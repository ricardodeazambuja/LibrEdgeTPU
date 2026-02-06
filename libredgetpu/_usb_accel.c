/*
 * _usb_accel — C extension for fast USB transfers to the Coral Edge TPU.
 *
 * Replaces pyusb with direct libusb-1.0 calls to eliminate:
 *   - array.array('B') conversion overhead on every write
 *   - ctypes marshaling
 *   - separate header + data USB writes (coalesced into one)
 *
 * Falls back gracefully: if this extension isn't compiled, transport.py
 * uses pyusb instead.
 *
 * Build: pip install -e "libredgetpu/" (requires libusb-1.0-dev)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <libusb.h>
#include <string.h>

/* USB endpoints (must match transport.py) */
#define EP_WRITE   0x01
#define EP_OUTPUT  0x81
#define EP_STATUS  0x82

/* Chunk size for bulk writes (1 MB, must match transport.py _CHUNK_SIZE) */
#define CHUNK_SIZE  0x100000

/* Default USB timeout in milliseconds */
#define DEFAULT_TIMEOUT_MS 6000

/* ── UsbDevice type ─────────────────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    libusb_context *ctx;
    libusb_device_handle *handle;
    int is_open;
    /* Reusable send buffer: 8 bytes header + CHUNK_SIZE data */
    unsigned char *send_buf;
} UsbDevice;

static void UsbDevice_dealloc(UsbDevice *self);
static PyObject *UsbDevice_close(UsbDevice *self, PyObject *Py_UNUSED(args));

/* ── Helper: raise from libusb error ──────────────────────────── */

static PyObject *
set_libusb_error(int rc, const char *context)
{
    PyErr_Format(PyExc_OSError, "%s: libusb error %d (%s)",
                 context, rc, libusb_strerror(rc));
    return NULL;
}

/* ── Constructor ──────────────────────────────────────────────── */

static int
UsbDevice_init(UsbDevice *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"vendor_id", "product_id", NULL};
    int vendor_id, product_id;
    int rc;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist,
                                     &vendor_id, &product_id))
        return -1;

    self->ctx = NULL;
    self->handle = NULL;
    self->is_open = 0;
    self->send_buf = NULL;

    rc = libusb_init(&self->ctx);
    if (rc < 0) {
        set_libusb_error(rc, "libusb_init");
        return -1;
    }

    self->handle = libusb_open_device_with_vid_pid(self->ctx,
                                                    (uint16_t)vendor_id,
                                                    (uint16_t)product_id);
    if (self->handle == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "No USB device found with given VID:PID");
        libusb_exit(self->ctx);
        self->ctx = NULL;
        return -1;
    }

    /* Detach kernel driver if active */
    if (libusb_kernel_driver_active(self->handle, 0) == 1) {
        libusb_detach_kernel_driver(self->handle, 0);
    }

    rc = libusb_set_configuration(self->handle, 1);
    if (rc < 0 && rc != LIBUSB_ERROR_BUSY) {
        set_libusb_error(rc, "set_configuration");
        libusb_close(self->handle);
        libusb_exit(self->ctx);
        self->handle = NULL;
        self->ctx = NULL;
        return -1;
    }

    rc = libusb_claim_interface(self->handle, 0);
    if (rc < 0) {
        set_libusb_error(rc, "claim_interface");
        libusb_close(self->handle);
        libusb_exit(self->ctx);
        self->handle = NULL;
        self->ctx = NULL;
        return -1;
    }

    /* Pre-allocate send buffer: header (8) + max chunk (CHUNK_SIZE) */
    self->send_buf = (unsigned char *)PyMem_Malloc(8 + CHUNK_SIZE);
    if (self->send_buf == NULL) {
        PyErr_NoMemory();
        libusb_release_interface(self->handle, 0);
        libusb_close(self->handle);
        libusb_exit(self->ctx);
        self->handle = NULL;
        self->ctx = NULL;
        return -1;
    }

    self->is_open = 1;
    return 0;
}

/* ── Destructor ───────────────────────────────────────────────── */

static void
UsbDevice_dealloc(UsbDevice *self)
{
    if (self->is_open) {
        /* Best-effort cleanup */
        UsbDevice_close(self, NULL);
    }
    if (self->send_buf) {
        PyMem_Free(self->send_buf);
        self->send_buf = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ── Ensure open guard ────────────────────────────────────────── */

static int
ensure_open(UsbDevice *self)
{
    if (!self->is_open) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Device not open -- call open() first or check USB connection");
        return 0;
    }
    return 1;
}

/* ── send(data, tag) ──────────────────────────────────────────── */

PyDoc_STRVAR(send_doc,
"send(data, tag)\n\n"
"Send framed bulk data with coalesced header.\n"
"The 8-byte header (length + tag) is prepended to the first chunk,\n"
"eliminating one USB round-trip compared to sending header separately.");

static PyObject *
UsbDevice_send(UsbDevice *self, PyObject *args)
{
    Py_buffer data_buf;
    int tag;
    int rc, transferred;
    uint32_t ll;
    unsigned char *data;
    unsigned char *sbuf;
    Py_ssize_t data_len;

    if (!ensure_open(self))
        return NULL;

    if (!PyArg_ParseTuple(args, "y*i", &data_buf, &tag))
        return NULL;

    data = (unsigned char *)data_buf.buf;
    data_len = data_buf.len;

    /* Bounds check: USB protocol header uses 32-bit length field */
    if (data_len < 0 || data_len > 0xFFFFFFFF) {
        PyBuffer_Release(&data_buf);
        PyErr_SetString(PyExc_ValueError,
                        "Data size exceeds 4GB USB protocol limit");
        return NULL;
    }

    ll = (uint32_t)data_len;
    sbuf = self->send_buf;

    /* Write header into send buffer */
    memcpy(sbuf, &ll, 4);     /* little-endian on LE platforms (x86/ARM) */
    memcpy(sbuf + 4, &tag, 4);

    if (data_len <= CHUNK_SIZE - 8) {
        /* Common fast path: header + all data in one USB write */
        memcpy(sbuf + 8, data, data_len);

        Py_BEGIN_ALLOW_THREADS
        rc = libusb_bulk_transfer(self->handle, EP_WRITE,
                                  sbuf, (int)(8 + data_len),
                                  &transferred, DEFAULT_TIMEOUT_MS);
        Py_END_ALLOW_THREADS

        PyBuffer_Release(&data_buf);
        if (rc < 0)
            return set_libusb_error(rc, "send (single chunk)");
    } else {
        /* Large data: header + first partial chunk, then remaining chunks */
        int first = CHUNK_SIZE - 8;
        memcpy(sbuf + 8, data, first);

        Py_BEGIN_ALLOW_THREADS
        rc = libusb_bulk_transfer(self->handle, EP_WRITE,
                                  sbuf, CHUNK_SIZE,
                                  &transferred, DEFAULT_TIMEOUT_MS);
        Py_END_ALLOW_THREADS

        if (rc < 0) {
            PyBuffer_Release(&data_buf);
            return set_libusb_error(rc, "send (first chunk)");
        }

        Py_ssize_t off = first;
        Py_ssize_t remaining = data_len - first;
        while (remaining > 0) {
            int chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : (int)remaining;

            Py_BEGIN_ALLOW_THREADS
            rc = libusb_bulk_transfer(self->handle, EP_WRITE,
                                      data + off, chunk,
                                      &transferred, DEFAULT_TIMEOUT_MS);
            Py_END_ALLOW_THREADS

            if (rc < 0) {
                PyBuffer_Release(&data_buf);
                return set_libusb_error(rc, "send (continuation)");
            }
            off += chunk;
            remaining -= chunk;
        }
        PyBuffer_Release(&data_buf);
    }

    Py_RETURN_NONE;
}

/* ── read_output(max_size, timeout_ms) ────────────────────────── */

PyDoc_STRVAR(read_output_doc,
"read_output(max_size=1024, timeout_ms=6000)\n\n"
"Read output tensor data from EP 0x81 into a pre-allocated buffer.\n"
"Returns bytes.");

static PyObject *
UsbDevice_read_output(UsbDevice *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"max_size", "timeout_ms", NULL};
    int max_size = 1024;
    int timeout_ms = DEFAULT_TIMEOUT_MS;
    int rc, transferred;
    unsigned char *buf;

    if (!ensure_open(self))
        return NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                     &max_size, &timeout_ms))
        return NULL;

    /* Sanity check max_size before arithmetic to prevent integer overflow */
    if (max_size < 0 || max_size > 100000000) {  /* 100 MB sanity limit */
        PyErr_Format(PyExc_ValueError,
                     "read_output max_size=%d out of valid range [0, 100000000]",
                     max_size);
        return NULL;
    }

    /* Allocate max_size + 32768 so chunked reads always have room for a
     * full 32768-byte chunk without overflow.  The returned data can be
     * LARGER than max_size — USB data boundaries don't align with DMA
     * hint output step sizes, and pyusb's read also returns variable
     * amounts.  execute_dma_hints() concatenates all output parts and
     * the total is correct. */
    int alloc_size = (max_size <= 32768) ? 32768 : (max_size + 32768);
    buf = (unsigned char *)PyMem_Malloc(alloc_size);
    if (buf == NULL)
        return PyErr_NoMemory();

    if (max_size <= 32768) {
        /* Single read — request max_size but buffer is 32768 to avoid
         * overflow from larger-than-expected device responses. */
        Py_BEGIN_ALLOW_THREADS
        rc = libusb_bulk_transfer(self->handle, EP_OUTPUT,
                                  buf, max_size,
                                  &transferred, timeout_ms);
        Py_END_ALLOW_THREADS

        if (rc == 0 || rc == LIBUSB_ERROR_OVERFLOW) {
            PyObject *result = PyBytes_FromStringAndSize((char *)buf, transferred);
            PyMem_Free(buf);
            return result;
        }
        PyMem_Free(buf);
        return set_libusb_error(rc, "read_output");
    }

    /* For large outputs, read in 32768-byte chunks.  Don't cap at
     * max_size — return ALL data received, matching pyusb behavior
     * where the bytearray can grow beyond max_size. */
    {
        int offset = 0;
        while (offset < max_size) {
            int remaining = alloc_size - offset;
            int request = (remaining > 32768) ? 32768 : remaining;
            if (request <= 0)
                break;

            Py_BEGIN_ALLOW_THREADS
            rc = libusb_bulk_transfer(self->handle, EP_OUTPUT,
                                      buf + offset, request,
                                      &transferred, timeout_ms);
            Py_END_ALLOW_THREADS

            if (rc == LIBUSB_ERROR_TIMEOUT)
                break;
            if (rc < 0 && rc != LIBUSB_ERROR_OVERFLOW) {
                PyMem_Free(buf);
                return set_libusb_error(rc, "read_output (chunk)");
            }
            if (transferred <= 0)
                break;
            offset += transferred;
        }

        PyObject *result = PyBytes_FromStringAndSize((char *)buf, offset);
        PyMem_Free(buf);
        return result;
    }
}

/* ── read_status(timeout_ms) ──────────────────────────────────── */

PyDoc_STRVAR(read_status_doc,
"read_status(timeout_ms=6000)\n\n"
"Read 16-byte status packet from EP 0x82. Returns bytes.");

static PyObject *
UsbDevice_read_status(UsbDevice *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"timeout_ms", NULL};
    int timeout_ms = DEFAULT_TIMEOUT_MS;
    int rc, transferred;
    unsigned char buf[16];

    if (!ensure_open(self))
        return NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &timeout_ms))
        return NULL;

    Py_BEGIN_ALLOW_THREADS
    rc = libusb_bulk_transfer(self->handle, EP_STATUS,
                              buf, 16,
                              &transferred, timeout_ms);
    Py_END_ALLOW_THREADS

    if (rc < 0)
        return set_libusb_error(rc, "read_status");

    return PyBytes_FromStringAndSize((char *)buf, transferred);
}

/* ── ctrl_transfer(bmRequestType, bRequest, wValue, wIndex, data_or_length) */

PyDoc_STRVAR(ctrl_transfer_doc,
"ctrl_transfer(bmRequestType, bRequest, wValue, wIndex, data_or_length)\n\n"
"Perform a USB control transfer. If data_or_length is bytes, it's an OUT\n"
"transfer. If it's an int, it's an IN transfer returning that many bytes.");

static PyObject *
UsbDevice_ctrl_transfer(UsbDevice *self, PyObject *args)
{
    int bmRequestType, bRequest, wValue, wIndex;
    PyObject *data_or_length;
    int rc;

    if (!ensure_open(self))
        return NULL;

    if (!PyArg_ParseTuple(args, "iiiiO", &bmRequestType, &bRequest,
                          &wValue, &wIndex, &data_or_length))
        return NULL;

    if (PyLong_Check(data_or_length)) {
        /* IN transfer */
        int length = (int)PyLong_AsLong(data_or_length);
        if (length < 0 || length > 65535) {
            PyErr_SetString(PyExc_ValueError,
                            "ctrl_transfer length must be in [0, 65535]");
            return NULL;
        }
        unsigned char *buf = (unsigned char *)PyMem_Malloc(length);
        if (buf == NULL)
            return PyErr_NoMemory();

        Py_BEGIN_ALLOW_THREADS
        rc = libusb_control_transfer(self->handle,
                                     (uint8_t)bmRequestType,
                                     (uint8_t)bRequest,
                                     (uint16_t)wValue,
                                     (uint16_t)wIndex,
                                     buf, (uint16_t)length,
                                     DEFAULT_TIMEOUT_MS);
        Py_END_ALLOW_THREADS

        if (rc < 0) {
            PyMem_Free(buf);
            return set_libusb_error(rc, "ctrl_transfer IN");
        }
        PyObject *result = PyBytes_FromStringAndSize((char *)buf, rc);
        PyMem_Free(buf);
        return result;
    } else {
        /* OUT transfer */
        Py_buffer data_buf;
        if (PyObject_GetBuffer(data_or_length, &data_buf, PyBUF_SIMPLE) < 0)
            return NULL;

        Py_BEGIN_ALLOW_THREADS
        rc = libusb_control_transfer(self->handle,
                                     (uint8_t)bmRequestType,
                                     (uint8_t)bRequest,
                                     (uint16_t)wValue,
                                     (uint16_t)wIndex,
                                     (unsigned char *)data_buf.buf,
                                     (uint16_t)data_buf.len,
                                     DEFAULT_TIMEOUT_MS);
        Py_END_ALLOW_THREADS

        PyBuffer_Release(&data_buf);
        if (rc < 0)
            return set_libusb_error(rc, "ctrl_transfer OUT");
        return PyLong_FromLong(rc);
    }
}

/* ── reset() ──────────────────────────────────────────────────── */

PyDoc_STRVAR(reset_doc, "reset()\n\nReset the USB device.");

static PyObject *
UsbDevice_reset(UsbDevice *self, PyObject *Py_UNUSED(args))
{
    int rc;

    if (!ensure_open(self))
        return NULL;

    Py_BEGIN_ALLOW_THREADS
    rc = libusb_reset_device(self->handle);
    Py_END_ALLOW_THREADS

    if (rc < 0)
        return set_libusb_error(rc, "reset");

    Py_RETURN_NONE;
}

/* ── close() ──────────────────────────────────────────────────── */

PyDoc_STRVAR(close_doc, "close()\n\nRelease the USB interface and close the device.");

static PyObject *
UsbDevice_close(UsbDevice *self, PyObject *Py_UNUSED(args))
{
    if (self->is_open && self->handle) {
        libusb_release_interface(self->handle, 0);
        libusb_close(self->handle);
        self->handle = NULL;
    }
    if (self->ctx) {
        libusb_exit(self->ctx);
        self->ctx = NULL;
    }
    self->is_open = 0;
    Py_RETURN_NONE;
}

/* ── is_kernel_driver_active(interface) ───────────────────────── */

PyDoc_STRVAR(is_kernel_driver_active_doc,
"is_kernel_driver_active(interface)\n\n"
"Check if a kernel driver is active on the given interface.");

static PyObject *
UsbDevice_is_kernel_driver_active(UsbDevice *self, PyObject *args)
{
    int interface_number;
    if (!ensure_open(self))
        return NULL;
    if (!PyArg_ParseTuple(args, "i", &interface_number))
        return NULL;
    int rc = libusb_kernel_driver_active(self->handle, interface_number);
    if (rc < 0)
        return set_libusb_error(rc, "is_kernel_driver_active");
    return PyBool_FromLong(rc);
}

/* ── Method table ─────────────────────────────────────────────── */

static PyMethodDef UsbDevice_methods[] = {
    {"send",          (PyCFunction)UsbDevice_send,
     METH_VARARGS, send_doc},
    {"read_output",   (PyCFunction)UsbDevice_read_output,
     METH_VARARGS | METH_KEYWORDS, read_output_doc},
    {"read_status",   (PyCFunction)UsbDevice_read_status,
     METH_VARARGS | METH_KEYWORDS, read_status_doc},
    {"ctrl_transfer", (PyCFunction)UsbDevice_ctrl_transfer,
     METH_VARARGS, ctrl_transfer_doc},
    {"reset",         (PyCFunction)UsbDevice_reset,
     METH_NOARGS, reset_doc},
    {"close",         (PyCFunction)UsbDevice_close,
     METH_NOARGS, close_doc},
    {"is_kernel_driver_active", (PyCFunction)UsbDevice_is_kernel_driver_active,
     METH_VARARGS, is_kernel_driver_active_doc},
    {NULL}
};

/* ── Type definition ──────────────────────────────────────────── */

static PyTypeObject UsbDeviceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "libredgetpu._usb_accel.UsbDevice",
    .tp_doc = "Fast USB device wrapper using direct libusb-1.0 calls.\n\n"
              "UsbDevice(vendor_id, product_id)\n\n"
              "Opens a USB device and claims interface 0.\n"
              "Methods: send(), read_output(), read_status(), "
              "ctrl_transfer(), reset(), close().\n\n"
              "Thread Safety: This class is NOT thread-safe. The Edge TPU\n"
              "hardware requires sequential command/response cycles, so\n"
              "concurrent USB operations would corrupt state at the hardware\n"
              "level anyway. Create separate instances per thread or add\n"
              "external synchronization if needed.",
    .tp_basicsize = sizeof(UsbDevice),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)UsbDevice_init,
    .tp_dealloc = (destructor)UsbDevice_dealloc,
    .tp_methods = UsbDevice_methods,
};

/* ── Module definition ────────────────────────────────────────── */

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name = "libredgetpu._usb_accel",
    .m_doc = "C extension for fast USB transfers to the Coral Edge TPU.\n"
             "Replaces pyusb with direct libusb-1.0 calls.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit__usb_accel(void)
{
    PyObject *m;

    if (PyType_Ready(&UsbDeviceType) < 0)
        return NULL;

    m = PyModule_Create(&module_def);
    if (m == NULL)
        return NULL;

    Py_INCREF(&UsbDeviceType);
    if (PyModule_AddObject(m, "UsbDevice", (PyObject *)&UsbDeviceType) < 0) {
        Py_DECREF(&UsbDeviceType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
