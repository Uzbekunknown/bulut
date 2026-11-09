/* input_logger.c - the mouse and the keys a game moves with, as raw input stamped with
 * QueryPerformanceCounter. The cloud simulator layer (cloud_sim_vk.c, sim_logger=1) starts it
 * beside the game in the second form below.
 *
 *   input_logger <out.txt> <stop event name, or -> <parent pid, or 0> [seconds, default 600]
 *   input_logger shm:<mapping name> <stop event name, or -> <parent pid, or 0> [seconds]
 *
 * The second form writes no file: it publishes the mouse alone (no keys) into the named file
 * mapping (mouse_shm.h), which the cloud simulator layer reads every frame. It runs until
 * the game ends (seconds up to a day). When the mapping has room for it (mouse_shm.h version 2), a
 * second thread also publishes the first connected XInput gamepad: polled about 1000 times a
 * second, a sample each time its state changes, and a "no pad" sample when it goes away.
 *
 * It stops when the event is set, when the parent process ends or after the seconds, whichever
 * comes first. Raw input is registered per process, so this runs apart from the game: a mouse
 * registered inside the game's process would take the game's own raw mouse away from it. With
 * RIDEV_INPUTSINK the window here, which is never shown, gets the events while the game is in
 * front. Only the keys a game moves with are written (W A S D Q E C, Space, Shift, Ctrl, Alt, the
 * arrows), so nothing typed elsewhere ends up in the file.
 *
 * out.txt, one event a line:
 *   # input_logger 1
 *   # frequency <QPC ticks a second>
 *   # start <QPC>
 *   m <QPC> <dx> <dy> <button flags, hex> <wheel> <device> <mouse flags, hex>
 *   k <QPC> <virtual key> <1 down, 0 up> <make code> <device>
 *   # end <QPC> <mouse events> <key events> <why: event, parent or time>
 * device: 0 for injected input (SendInput), else 1, 2, ... in the order the devices were first
 * seen. The QPC is taken as the event is read, normally well under a millisecond after it came. */
#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <xinput.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include "mouse_shm.h"
#pragma comment(lib, "xinput.lib")

static FILE *out;
static struct mouse_shm *shm;   /* the shm: form */
static int shm_pad;             /* ... and its mapping has the pad part */
static volatile LONG pad_stop;
static unsigned long long mice, keys, pads;
static HANDLE device[32];
static int devices;

static int device_id(HANDLE h)
{
	if (!h) return 0;
	for (int i = 0; i < devices; i++)
		if (device[i] == h) return i + 1;
	if (devices < 32) {
		device[devices++] = h;
		return devices;
	}
	return 99;
}

static int game_key(USHORT vk)
{
	switch (vk) {
	case 'W': case 'A': case 'S': case 'D': case 'Q': case 'E': case 'C':
	case VK_SPACE: case VK_SHIFT: case VK_CONTROL: case VK_MENU:
	case VK_LSHIFT: case VK_RSHIFT: case VK_LCONTROL: case VK_RCONTROL: case VK_LMENU: case VK_RMENU:
	case VK_UP: case VK_DOWN: case VK_LEFT: case VK_RIGHT:
		return 1;
	default:
		return 0;
	}
}

static void got(HRAWINPUT handle)
{
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	RAWINPUT ri;
	UINT size = sizeof ri;
	if (GetRawInputData(handle, RID_INPUT, &ri, &size, sizeof(RAWINPUTHEADER)) == (UINT)-1) return;
	int id = device_id(ri.header.hDevice);
	if (shm) {
		if (ri.header.dwType != RIM_TYPEMOUSE || (ri.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)) return;
		if (!ri.data.mouse.lLastX && !ri.data.mouse.lLastY) return;
		struct mouse_shm_event *e = &shm->ring[shm->count & (MOUSE_SHM_EVENTS - 1)];
		e->qpc = now.QuadPart;
		e->dx = ri.data.mouse.lLastX;
		e->dy = ri.data.mouse.lLastY;
		InterlockedIncrement64((volatile LONG64 *)&shm->count);
		mice++;
		return;
	}
	if (ri.header.dwType == RIM_TYPEMOUSE) {
		const RAWMOUSE *m = &ri.data.mouse;
		int wheel = (m->usButtonFlags & (RI_MOUSE_WHEEL | RI_MOUSE_HWHEEL)) ? (int)(SHORT)m->usButtonData : 0;
		fprintf(out, "m %lld %ld %ld %x %d %d %x\n", now.QuadPart, m->lLastX, m->lLastY, (unsigned)m->usButtonFlags,
			wheel, id, (unsigned)m->usFlags);
		mice++;
	} else if (ri.header.dwType == RIM_TYPEKEYBOARD) {
		const RAWKEYBOARD *k = &ri.data.keyboard;
		if (!game_key(k->VKey)) return;
		fprintf(out, "k %lld %u %d %u %d\n", now.QuadPart, (unsigned)k->VKey, (k->Flags & RI_KEY_BREAK) ? 0 : 1,
			(unsigned)k->MakeCode, id);
		keys++;
	}
}

static void pad_put(const XINPUT_GAMEPAD *g, LONGLONG qpc, DWORD packet, uint32_t user)
{
	struct mouse_shm_pad *e = &shm->pads[shm->pad_count & (MOUSE_SHM_PADS - 1)];
	e->qpc = qpc;
	e->lx = g ? g->sThumbLX : 0, e->ly = g ? g->sThumbLY : 0;
	e->rx = g ? g->sThumbRX : 0, e->ry = g ? g->sThumbRY : 0;
	e->lt = g ? g->bLeftTrigger : 0, e->rt = g ? g->bRightTrigger : 0;
	e->buttons = g ? g->wButtons : 0;
	e->packet = packet;
	e->user = user;
	InterlockedIncrement64((volatile LONG64 *)&shm->pad_count);
	pads++;
}

/* The gamepad: the first connected XInput user, looked for once a second while there is none
 * (XInputGetState on an empty slot is slow), then read about every millisecond. */
static DWORD WINAPI pad_thread(LPVOID unused)
{
	(void)unused;
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	HANDLE timer = CreateWaitableTimerExW(NULL, NULL, 0x00000002 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */, TIMER_ALL_ACCESS);
	if (!timer) timer = CreateWaitableTimerW(NULL, FALSE, NULL);
	int user = -1;
	DWORD last = 0;
	ULONGLONG scanned = 0;
	while (!pad_stop) {
		XINPUT_STATE st;
		LARGE_INTEGER now;
		if (user < 0 && GetTickCount64() - scanned >= 1000) {
			scanned = GetTickCount64();
			for (DWORD u = 0; u < XUSER_MAX_COUNT && user < 0; u++) {
				if (XInputGetState(u, &st) != ERROR_SUCCESS) continue;
				QueryPerformanceCounter(&now);
				user = (int)u, last = st.dwPacketNumber;
				shm->pad_connected = 1;
				pad_put(&st.Gamepad, now.QuadPart, st.dwPacketNumber, u);
			}
		}
		if (user >= 0) {
			DWORD r = XInputGetState((DWORD)user, &st);
			QueryPerformanceCounter(&now);
			if (r != ERROR_SUCCESS) {
				shm->pad_connected = 0;
				pad_put(NULL, now.QuadPart, 0, MOUSE_SHM_NO_PAD);
				user = -1;
			} else if (st.dwPacketNumber != last) {
				last = st.dwPacketNumber;
				pad_put(&st.Gamepad, now.QuadPart, st.dwPacketNumber, (uint32_t)user);
			}
		}
		LARGE_INTEGER due;
		due.QuadPart = -10000;               /* 1 ms, relative */
		if (timer && SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE)) WaitForSingleObject(timer, 20);
		else Sleep(1);
	}
	if (timer) CloseHandle(timer);
	return 0;
}

static LRESULT CALLBACK proc(HWND wnd, UINT msg, WPARAM wp, LPARAM lp)
{
	if (msg == WM_INPUT) got((HRAWINPUT)lp);
	return DefWindowProcW(wnd, msg, wp, lp);   /* WM_INPUT too: it frees the event */
}

static void pump(void)
{
	MSG msg;
	while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) DispatchMessageW(&msg);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command, int show)
{
	(void)previous;
	(void)command;
	(void)show;
	int argc = 0;
	wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	if (!argv || argc < 4) return 2;
	HANDLE mapping = NULL;
	if (!wcsncmp(argv[1], L"shm:", 4)) {
		mapping = OpenFileMappingW(FILE_MAP_WRITE | FILE_MAP_READ, FALSE, argv[1] + 4);
		shm = mapping ? MapViewOfFile(mapping, FILE_MAP_WRITE | FILE_MAP_READ, 0, 0, sizeof *shm) : NULL;
		shm_pad = shm != NULL;
		if (!shm && mapping)                   /* a version 1 mapping: the mouse alone */
			shm = MapViewOfFile(mapping, FILE_MAP_WRITE | FILE_MAP_READ, 0, 0, MOUSE_SHM_V1_BYTES);
		if (!shm) return 5;
		out = _wfopen(L"NUL", L"wb");     /* the log lines below go nowhere */
	} else {
		out = _wfopen(argv[1], L"wb");
	}
	if (!out) return 3;
	setvbuf(out, NULL, _IOFBF, 1 << 16);
	int named = wcscmp(argv[2], L"-") != 0;
	HANDLE stop = named ? OpenEventW(SYNCHRONIZE, FALSE, argv[2]) : NULL;
	DWORD stop_error = GetLastError();
	DWORD pid = (DWORD)wcstoul(argv[3], NULL, 10);
	HANDLE parent = pid ? OpenProcess(SYNCHRONIZE, FALSE, pid) : NULL;
	DWORD parent_error = GetLastError();
	double seconds = argc > 4 ? _wtof(argv[4]) : 600.0;
	if (!(seconds > 0.0) || seconds > (shm ? 86400.0 : 3600.0)) seconds = 600.0;

	LARGE_INTEGER frequency, start;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);
	HANDLE pad = NULL;
	if (shm) {
		shm->frequency = frequency.QuadPart;
		shm->version = shm_pad ? MOUSE_SHM_VERSION : 1;
		shm->writer_pid = (int32_t)GetCurrentProcessId();
		if (shm_pad) pad = CreateThread(NULL, 0, pad_thread, NULL, 0, NULL);
	}
	fprintf(out, "# input_logger 1\n# frequency %lld\n# start %lld\n", frequency.QuadPart, start.QuadPart);
	if (named && !stop) fprintf(out, "# no stop event %ls (error %lu)\n", argv[2], stop_error);
	if (pid && !parent) fprintf(out, "# no parent process %lu (error %lu)\n", pid, parent_error);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

	WNDCLASSW wc = { 0 };
	wc.lpfnWndProc = proc;
	wc.hInstance = instance;
	wc.lpszClassName = L"input_logger";
	RegisterClassW(&wc);
	HWND wnd = CreateWindowExW(0, wc.lpszClassName, L"input_logger", 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, instance, NULL);
	RAWINPUTDEVICE rid[2] = { { 0x01, 0x02, RIDEV_INPUTSINK, wnd }, { 0x01, 0x06, RIDEV_INPUTSINK, wnd } };
	UINT kinds = shm ? 1 : 2;                  /* the shm: form takes the mouse alone */
	if (!wnd || !RegisterRawInputDevices(rid, kinds, sizeof rid[0])) {
		fprintf(out, "# end %lld 0 0 no-raw-input (error %lu)\n", start.QuadPart, GetLastError());
		fclose(out);
		return 4;
	}

	HANDLE waits[2];
	DWORD n = 0;
	if (stop) waits[n++] = stop;
	if (parent) waits[n++] = parent;
	ULONGLONG until = GetTickCount64() + (ULONGLONG)(seconds * 1000.0), flushed = GetTickCount64();
	const char *why = "time";
	for (;;) {
		pump();
		ULONGLONG now = GetTickCount64();
		if (now - flushed >= 250) {
			fflush(out);
			flushed = now;
		}
		if (now >= until) break;
		DWORD wait = until - now < 250 ? (DWORD)(until - now) : 250;
		DWORD r = MsgWaitForMultipleObjects(n, waits, FALSE, wait, QS_ALLINPUT);
		if (r == WAIT_FAILED) {
			why = "wait-failed";
			break;
		}
		if (r < WAIT_OBJECT_0 + n) {
			why = waits[r - WAIT_OBJECT_0] == stop ? "event" : "parent";
			break;
		}
	}
	pump();                                    /* what came in before the stop */
	if (pad) {
		InterlockedExchange(&pad_stop, 1);
		WaitForSingleObject(pad, 1000);
		CloseHandle(pad);
	}
	LARGE_INTEGER end;
	QueryPerformanceCounter(&end);
	fprintf(out, "# end %lld %llu %llu %s\n", end.QuadPart, mice, keys, why);
	fclose(out);
	rid[0].dwFlags = rid[1].dwFlags = RIDEV_REMOVE;
	rid[0].hwndTarget = rid[1].hwndTarget = NULL;
	RegisterRawInputDevices(rid, kinds, sizeof rid[0]);
	if (shm) UnmapViewOfFile(shm);
	if (mapping) CloseHandle(mapping);
	return 0;
}
