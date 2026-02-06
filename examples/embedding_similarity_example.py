#!/usr/bin/env python3
"""Embedding similarity example — place recognition / similarity search.

Builds a gallery of scene embeddings and finds the closest match in
real-time using Edge TPU-accelerated cosine similarity (MatMulEngine).

Press 's' to snapshot the current frame into the gallery.
Press 'c' to clear the gallery.

Requirements: Edge TPU USB accelerator, opencv-python
"""

import argparse
import time

import cv2
import numpy as np

from libredgetpu import EmbeddingSimilarity
from _common import add_common_args, WebcamLoop, draw_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="EmbeddingSimilarity — place recognition")
    add_common_args(parser)
    parser.add_argument("--dim", type=int, default=256,
                        choices=[256, 512, 1024],
                        help="Embedding dimension (default: 256)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of results to return (default: 3)")
    parser.add_argument("--max-gallery", type=int, default=10,
                        help="Max gallery items (default: 10)")
    parser.add_argument("--save-db", type=str, default=None,
                        help="Path to save gallery as .npz on exit")
    parser.add_argument("--load-db", type=str, default=None,
                        help="Path to load gallery from .npz on start")
    return parser.parse_args()


def frame_to_embedding(frame, dim):
    """Generate a pseudo-embedding from frame spatial statistics.

    Divides the frame into an 8x8 grid, computes mean RGB per cell,
    and L2-normalizes the result.  Matches the approach used in
    the GUI (algorithm_modes.py).
    """
    h, w = frame.shape[:2]
    grid_h, grid_w = 8, 8
    cell_h, cell_w = h // grid_h, w // grid_w
    features = []
    for i in range(grid_h):
        for j in range(grid_w):
            cell = frame[i * cell_h:(i + 1) * cell_h,
                         j * cell_w:(j + 1) * cell_w]
            features.append(cell.mean(axis=(0, 1)))  # [3] BGR means
    features = np.concatenate(features).astype(np.float32)  # [192]

    emb = np.zeros(dim, dtype=np.float32)
    n = min(len(features), dim)
    emb[:n] = features[:n]

    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb /= norm
    return emb


def main():
    args = parse_args()
    loop = WebcamLoop(args)

    gallery_count = 0
    thumbnails = {}  # label -> thumbnail

    with EmbeddingSimilarity.from_template(args.dim) as sim:
        # Load existing gallery
        if args.load_db:
            data = np.load(args.load_db)
            for i, (label, emb) in enumerate(
                    zip(data["labels"], data["embeddings"])):
                label = str(label)
                sim.add(label, emb)
                gallery_count += 1
                # No thumbnail for loaded items
                thumbnails[label] = None
            print(f"Loaded {gallery_count} items from {args.load_db}")

        save_embeddings = []
        save_labels = []

        for frame in loop:
            query = frame_to_embedding(frame, args.dim)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            # Add to gallery on 's'
            if key == ord("s") and gallery_count < args.max_gallery:
                label = f"scene_{gallery_count}"
                sim.add(label, query)
                thumbnails[label] = cv2.resize(frame, (64, 48))
                save_embeddings.append(query.copy())
                save_labels.append(label)
                gallery_count += 1
                print(f"Added '{label}' to gallery ({gallery_count}/{args.max_gallery})")

            # Clear gallery on 'c'
            if key == ord("c"):
                sim.clear()
                gallery_count = 0
                thumbnails.clear()
                save_embeddings.clear()
                save_labels.clear()
                print("Gallery cleared")

            h, w = frame.shape[:2]

            if gallery_count > 0:
                t0 = time.perf_counter()
                results = sim.query(query, top_k=args.top_k)
                latency_ms = (time.perf_counter() - t0) * 1000

                # Draw top-k results
                for i, (label, score) in enumerate(results):
                    thumb = thumbnails.get(label)
                    y_off = 100 + i * 60
                    if thumb is not None and y_off + 48 < h and 74 < w:
                        frame[y_off:y_off + 48, 10:74] = thumb
                        cv2.rectangle(frame, (9, y_off - 1), (75, y_off + 49),
                                      (0, 255, 255), 1)
                    draw_text(frame, f"#{i+1} {label}: {score:.3f}",
                              (80, y_off + 25))

                draw_text(frame,
                          f"Gallery: {gallery_count}/{args.max_gallery} | {latency_ms:.1f} ms",
                          (10, 30))
                loop.print_metrics(
                    {"gallery": gallery_count,
                     "top1": results[0][1] if results else 0},
                    latency_ms)
            else:
                latency_ms = 0.0
                draw_text(frame, "Press 's' to add scene to gallery", (10, 30))
                loop.print_metrics({"gallery": 0}, 0.0)

            draw_text(frame, "'s'=snapshot  'c'=clear  'q'=quit", (10, h - 20))
            loop.show(frame)

        # Save gallery on exit
        if args.save_db and save_embeddings:
            np.savez(args.save_db,
                     embeddings=np.array(save_embeddings),
                     labels=np.array(save_labels))
            print(f"Saved {len(save_embeddings)} embeddings to {args.save_db}")

    loop.cleanup()


if __name__ == "__main__":
    main()
