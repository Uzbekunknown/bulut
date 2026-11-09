/* mouse_shm.h - the mouse as input_logger publishes it for the cloud simulator layer (cloud_sim_vk.c):
 * a ring of raw mouse events in a named file mapping, one writer (input_logger), any readers.
 * The writer fills ring[count % MOUSE_SHM_EVENTS], then increments count (a full barrier), so a
 * reader that sees count = c may read events c-1, c-2, ... back to c - MOUSE_SHM_EVENTS + 1.
 *
 * Version 2 adds the first connected XInput gamepad after the mouse, in the same way: input_logger
 * polls it about 1000 times a second and writes a sample each time its state changes (XInput's
 * packet number), so between two samples the sticks held the earlier one's values. A mapping made
 * for version 1 (MOUSE_SHM_V1_BYTES) has no pad part; input_logger then writes the mouse alone. */
#ifndef MOUSE_SHM_H
#define MOUSE_SHM_H
#include <stddef.h>
#include <stdint.h>

#define MOUSE_SHM_EVENTS 65536          /* a power of two: ~65 s at 1000 Hz */
#define MOUSE_SHM_PADS 32768            /* a power of two: ~130 s at 250 changes a second */
#define MOUSE_SHM_VERSION 2

struct mouse_shm_event {
	int64_t qpc;                     /* QueryPerformanceCounter as the event was read */
	int32_t dx, dy;                  /* raw counts, before any pointer acceleration */
};

struct mouse_shm_pad {
	int64_t qpc;                     /* QueryPerformanceCounter as the state was read */
	int16_t lx, ly, rx, ry;          /* the sticks, XInput's raw -32768..32767 (y up is positive) */
	uint8_t lt, rt;                  /* the triggers, 0..255 */
	uint16_t buttons;                /* XINPUT_GAMEPAD_* bits */
	uint32_t packet;                 /* XInput's packet number */
	uint32_t user;                   /* the XInput user index 0..3, or MOUSE_SHM_NO_PAD: the pad went away
					    (the sticks then read 0) */
};
#define MOUSE_SHM_NO_PAD 0xFFFFFFFFu

struct mouse_shm {
	volatile int64_t count;          /* events written so far */
	int64_t frequency;               /* QPC ticks a second */
	int32_t version, writer_pid;
	struct mouse_shm_event ring[MOUSE_SHM_EVENTS];
	/* version 2 */
	volatile int64_t pad_count;      /* pad samples written so far */
	int32_t pad_connected, pad_reserved;
	struct mouse_shm_pad pads[MOUSE_SHM_PADS];
};

#define MOUSE_SHM_V1_BYTES offsetof(struct mouse_shm, pad_count)

#endif
