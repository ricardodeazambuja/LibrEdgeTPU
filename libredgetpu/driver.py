"""Hardware initialization and execution protocol for the Edge TPU.

Translates the register-write sequences from libedgetpu's C++ source into
a clean driver API.  The init sequence follows libedgetpu's phases:
Open, EnableReset, QuitReset, EnableHardwareClockGate, InitializeChip,
DoRunControl, RegisterAndEnableAllInterrupts.
"""

import struct

from .transport import USBTransport, TAG_INSTRUCTIONS, TAG_INPUT_ACTIVATIONS, TAG_PARAMETERS


def _q(x):
    return struct.pack("<Q", x)


def _le32(x):
    return struct.pack("<I", x)


def _read_le32(raw):
    return struct.unpack("<I", raw[:4])[0]


# scu_ctrl_3 bitfield helpers
_SCU3_CUR_PWR_STATE_MASK = 0x3 << 8        # bits [9:8]
_SCU3_CUR_PWR_STATE_SLEEPING = 0x2 << 8
_SCU3_CUR_PWR_STATE_RUNNING = 0x0 << 8
_SCU3_RG_FORCE_SLEEP_MASK = 0x3 << 22      # bits [23:22]
_SCU3_RG_FORCE_SLEEP_SLEEP = 0x3 << 22     # 0b11 = force sleep
_SCU3_RG_FORCE_SLEEP_EXIT = 0x2 << 22      # 0b10 = exit sleep
_SCU3_RG_GCB_CLKDIV_MASK = 0x3 << 20       # bits [21:20]
_SCU3_RG_AXI_CLK_125M_MASK = 0x1 << 16     # bit 16
_SCU3_RG_8051_CLK_250M_MASK = 0x1 << 17    # bit 17


class EdgeTPUDriver:
    """Mid-level driver: init hardware, run cached or standalone executables."""

    def __init__(self, transport: USBTransport) -> None:
        self._t = transport
        self._cached_token: int = 0  # currently cached parameter_caching_token

    # ── Delegation methods (avoid reaching into _t from invokers) ─────────

    def send_raw(self, data: bytes, tag: int) -> None:
        """Send framed bulk data with the given tag (delegates to transport)."""
        self._t.send(data, tag)

    def read_status_packet(self) -> bytes:
        """Read a status packet from the device (delegates to transport)."""
        return self._t.read_status()

    # ── Parameter caching ─────────────────────────────────────────────────

    def reset_cached_parameters(self) -> None:
        """Invalidate cached parameter state, forcing a full param upload next time."""
        self._cached_token = 0

    def init_hardware(self) -> None:
        """Perform the full hardware initialization sequence.

        Follows libedgetpu's phases: Open, EnableReset, QuitReset (MAX perf,
        USB), EnableHardwareClockGate, InitializeChip, DoRunControl,
        RegisterAndEnableAllInterrupts, plus misc registers from USB traces.
        """
        t = self._t
        self._cached_token = 0

        # ── Phase 1: Open ──
        # Read scu_ctrl_0, clear USB inactive PHY mode bits [13:11] to 0, write back
        raw = t.read_register("scu_ctrl_0", 4)
        val = _read_le32(raw)
        val &= ~(0x7 << 11)  # clear bits [13:11]
        t.write_register("scu_ctrl_0", _le32(val))
        # Read scu_ctrl_2 to detect clock gate state
        t.read_register("scu_ctrl_2", 4)

        # ── Phase 2: EnableReset ──
        # Set rg_force_sleep bits [23:22] = 0b11 to force sleep
        raw = t.read_register("scu_ctrl_3", 4)
        val = _read_le32(raw)
        val = (val & ~_SCU3_RG_FORCE_SLEEP_MASK) | _SCU3_RG_FORCE_SLEEP_SLEEP
        t.write_register("scu_ctrl_3", _le32(val))

        # Poll scu_ctrl_3 until cur_pwr_state bits [9:8] == 0x2 (sleeping)
        t.poll_register("scu_ctrl_3", _SCU3_CUR_PWR_STATE_SLEEPING,
                        mask=_SCU3_CUR_PWR_STATE_MASK, read_len=4)

        # Pulse gcbb_credit0 (write 0xF then 0x0)
        t.write_register("gcbb_credit0", _le32(0xF))
        t.write_register("gcbb_credit0", _le32(0x0))

        # ── Phase 3: QuitReset (MAX performance, USB) ──
        raw = t.read_register("scu_ctrl_3", 4)
        val = _read_le32(raw)
        # rg_force_sleep = 0b10 (exit sleep)
        val = (val & ~_SCU3_RG_FORCE_SLEEP_MASK) | _SCU3_RG_FORCE_SLEEP_EXIT
        # rg_gcb_clkdiv = 0 (500 MHz)
        val &= ~_SCU3_RG_GCB_CLKDIV_MASK
        # rg_axi_clk_125m = 0 (250 MHz AXI)
        val &= ~_SCU3_RG_AXI_CLK_125M_MASK
        # rg_8051_clk_250m = 0 (500 MHz 8051)
        val &= ~_SCU3_RG_8051_CLK_250M_MASK
        t.write_register("scu_ctrl_3", _le32(val))

        # Poll scu_ctrl_3 until cur_pwr_state == 0x0 (running)
        t.poll_register("scu_ctrl_3", _SCU3_CUR_PWR_STATE_RUNNING,
                        mask=_SCU3_CUR_PWR_STATE_MASK, read_len=4)

        # Poll scalarCoreRunControl until == 0 (reset complete)
        t.poll_register("scalarCoreRunControl", 0, read_len=8)

        # Idle register: counter=1, idle enabled
        t.write_register("idleRegister", _q(1))

        # Tile config: broadcast all tiles (0x7F), poll until readback matches
        t.write_register("tileconfig0", _q(0x7F))
        t.poll_register("tileconfig0", 0x7F, read_len=8)

        # Deep sleep timing: toSleepDelay=2, toWakeDelay=30
        t.write_register("deepSleep", _q(0x001E02))

        # ── Phase 4: EnableHardwareClockGate ──
        raw = t.read_register("scu_ctrl_2", 4)
        val = _read_le32(raw)
        # Set rg_gated_gcb bits [19:18] = 0b01 (enable gated clock)
        val = (val & ~(0x3 << 18)) | (0x1 << 18)
        t.write_register("scu_ctrl_2", _le32(val))

        # ── Phase 5: InitializeChip ──
        t.read_register("omc0_00", 4)  # read e-fuse
        t.write_register("descr_ep", _q(0xF0))
        t.write_register("multi_bo_ep", _q(0))
        t.write_register("outfeed_chunk_length", _q(0x80))

        # ── Phase 6: DoRunControl (kMoveToRun) ──
        t.write_register("scalarCoreRunControl", _q(1))
        t.write_register("avDataPopRunControl", _q(1))
        t.write_register("parameterPopRunControl", _q(1))
        t.write_register("infeedRunControl", _q(1))
        t.write_register("outfeedRunControl", _q(1))
        t.write_register("opRunControl", _q(1))
        t.write_register("narrowToWideRunControl", _q(1))
        t.write_register("wideToNarrowRunControl", _q(1))
        for i in range(4):
            t.write_register(f"meshBus{i}RunControl", _q(1))
        t.write_register("ringBusConsumer0RunControl", _q(1))
        t.write_register("ringBusConsumer1RunControl", _q(1))
        t.write_register("ringBusProducerRunControl", _q(1))

        # ── Phase 7: RegisterAndEnableAllInterrupts ──
        t.write_register("fatal_err_int_control", _q(1))
        for i in range(4):
            t.write_register(f"top_level_int_{i}_control", _q(1))

        # ── Phase 8: Misc hardware config (from USB trace, not in libedgetpu open-source) ──
        t.read_register("omc0_d4", 4)
        t.write_register("omc0_d4", b"\x01\x00\x00\x80")
        t.read_register("rambist_ctrl_1", 4)
        t.write_register("rambist_ctrl_1", b"\x7f\x00\x00\x00")
        t.read_register("scu_ctr_7", 4)
        t.write_register("scu_ctr_7", b"\x3f\x00\x00\x00")

        # PCI-E / ABM (needed even on USB)
        t.write_register("slv_abm_en", b"\x01\x00\x00\x00")
        t.write_register("mst_abm_en", b"\x01\x00\x00\x00")
        t.write_register("slv_err_resp_isr_mask", b"\x03\x00\x00\x00")
        t.write_register("mst_err_resp_isr_mask", b"\x03\x00\x00\x00")

        t.read_register("omc0_d8", 4)
        t.write_register("omc0_d8", b"\x00\x00\x00\x80")

    def execute_cached(self, pc_instructions: bytes, params: bytes,
                       eo_instructions: bytes, input_data: bytes,
                       output_size: int) -> bytes:
        """Run a cached-mode model (weights <= ~8 MB).

        1. Send PARAMETER_CACHING instructions + parameters, wait status
        2. Send EXECUTION_ONLY instructions + input activations
        3. Read output (may arrive in chunks before status), then read status

        Returns output bytes.
        """
        t = self._t

        # Phase 1: cache parameters
        t.send(pc_instructions, TAG_INSTRUCTIONS)
        t.send(params, TAG_PARAMETERS)
        t.read_status()

        # Phase 2: execute with cached parameters
        t.send(eo_instructions, TAG_INSTRUCTIONS)
        t.send(input_data, TAG_INPUT_ACTIVATIONS)

        # Output arrives before status for large models
        output = t.read_output(max_size=output_size)
        t.read_status()

        return output

    def execute_inference_only(self, eo_instructions: bytes, input_data: bytes,
                              output_size: int) -> bytes:
        """Run inference using already-cached parameters (EXECUTION_ONLY phase only).

        Skips the parameter upload entirely -- only sends EXECUTION_ONLY
        instructions + input activations.  Parameters must have been cached
        by a prior execute_cached() call.

        Returns output bytes.
        """
        t = self._t

        t.send(eo_instructions, TAG_INSTRUCTIONS)
        t.send(input_data, TAG_INPUT_ACTIVATIONS)

        output = t.read_output(max_size=output_size)
        t.read_status()

        return output

    def execute_dma_hints(self, dma_steps, bitstreams, input_data: bytes,
                          params: bytes = None, output_size: int = 0) -> bytes:
        """Execute following the DMA hint sequence from the DarwiNN executable.

        This is the general-purpose execution path that handles multi-chunk
        instruction streams and split-input models (e.g., PoseNet, DeepLabV3).

        Args:
            dma_steps: List of DmaStep from the parsed executable.
            bitstreams: List of instruction bitstream bytes (one per chunk).
            input_data: Raw input bytes.
            params: Raw parameter bytes (for standalone/param-caching phases).
            output_size: Expected total output size in bytes.

        Returns output bytes.
        """
        t = self._t
        output_parts = []

        # Pad input to cover the full DMA range — the hardware may read
        # slightly past the actual tensor data for alignment (see libedgetpu
        # dma_info_extractor.cc: "DMA may request a small amount of data
        # past the end of the input buffer").
        if input_data:
            max_input = max(
                (s.offset + s.size for s in dma_steps if s.kind == "input"),
                default=0,
            )
            if max_input > len(input_data):
                padded = bytearray(max_input)
                padded[:len(input_data)] = input_data
                input_data = bytes(padded)

        for step in dma_steps:
            if step.kind == "instruction":
                if step.chunk_index < 0 or step.chunk_index >= len(bitstreams):
                    raise ValueError(
                        f"DMA instruction chunk_index={step.chunk_index} "
                        f"out of range (have {len(bitstreams)} bitstreams)"
                    )
                t.send(bitstreams[step.chunk_index], TAG_INSTRUCTIONS)
            elif step.kind == "input":
                if step.offset + step.size > len(input_data):
                    raise ValueError(
                        f"DMA input step exceeds buffer: "
                        f"offset={step.offset}, size={step.size}, "
                        f"buffer={len(input_data)}"
                    )
                chunk = input_data[step.offset:step.offset + step.size]
                t.send(chunk, TAG_INPUT_ACTIVATIONS)
            elif step.kind == "parameter":
                if params is not None:
                    if step.offset + step.size > len(params):
                        raise ValueError(
                            f"DMA parameter step exceeds buffer: "
                            f"offset={step.offset}, size={step.size}, "
                            f"buffer={len(params)}"
                        )
                    chunk = params[step.offset:step.offset + step.size]
                    t.send(chunk, TAG_PARAMETERS)
            elif step.kind == "output":
                out = t.read_output(max_size=step.size)
                output_parts.append(out)
            elif step.kind == "interrupt":
                t.read_status()
            elif step.kind == "fence":
                pass  # fences are implicit in our synchronous USB protocol

        return b"".join(output_parts)

    def execute_standalone(self, instructions: bytes, params: bytes,
                           input_data: bytes, output_size: int) -> bytes:
        """Run a standalone/streamed model (weights > ~8 MB).

        Sends instructions, input, and parameters in a single pass.
        Returns output bytes.
        """
        t = self._t

        t.send(instructions, TAG_INSTRUCTIONS)
        t.send(input_data, TAG_INPUT_ACTIVATIONS)
        t.send(params, TAG_PARAMETERS)

        output = t.read_output(max_size=output_size)
        t.read_status()

        return output
