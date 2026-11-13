/* cloud_sim_vk.c - the cloud gaming simulator as a Vulkan layer, VK_LAYER_UZ_cloud_sim.
 *
 * It works at the present, so it needs no upscaler or engine hook in the game (DOOM (2016) has
 * none): the frame the game presents goes into a ring on the GPU with the game's depth buffer, and
 * the screen shows, by mode:
 *   local  the newest frame: the game as it is
 *   cloud  the newest frame at least `latency` old: what a cloud player sees today
 *   warp   that old frame turned to where the camera is now, by the mouse alone. In first person the
 *          camera turns about the eye, so a rotation moves every pixel the same way whatever its
 *          depth (a homography): each screen pixel looks up where it was. Depth tells the weapon
 *          from the world: pixels nearer than sim_weapon_depth (and the sim_lock rectangles: the
 *          HUD) stay where they are on the screen, and what the weapon hid becomes a hole, filled
 *          from the farthest surface around it (a push-pull pyramid).
 * In every mode the screen shows the central `keep` of the frame, and the game's crosshair (a box
 * at the centre) is replaced by the layer's own, so the three modes look alike.
 *
 * The depth buffer is found the way capture tools find it: every depth-format image used as a
 * render pass's depth attachment is followed through its framebuffers, its draws are counted per
 * frame, and its layout is followed through render passes and barriers in submission order. At
 * the present the one with the most draws this frame, of the screen's shape, is copied.
 *
 * Settings: cloud-sim-vk.txt beside the DLL, or the file CLOUD_SIM_VK_SETTINGS names; read at the
 * start and again whenever it changes (sim_mode, sim_latency, sim_record_now, sim_snapshot_now act
 * when their value in the file changes, so F6/F7 choices survive other edits):
 *   sim=1  sim_mode=0|1|2  sim_latency=<ms>  sim_keep=<share>  sim_slots=<frames, at start>
 *   sim_cx=<x y z> sim_cy=<x y z>  rotation vector per mouse count in x / y (radians; camera axes
 *                                 x right, y down, z forward; P_new = R P_old)
 *   sim_tau_ms=<ms>     the game's delay from the mouse to the present
 *   sim_lag_ms=<ms>     the camera's first-order lag behind the mouse
 *   sim_hfov=<deg>      the horizontal field of view of the presented frame
 *   sim_depth=0|1       follow and copy the depth buffer (at start)
 *   sim_depth_reversed=auto|0|1
 *   sim_weapon_depth=<raw depth>  nearer than this stays on the screen (0: nothing)
 *   sim_lock1..4=<x0 y0 x1 y1>    screen-locked rectangles, shares of the screen
 *   sim_hole=<share of height>    half-size of the box at the centre left out (the game's crosshair)
 *   sim_cross=<share of height>   the arm of the layer's crosshair
 *   sim_logger=0|1      start input_logger.exe (beside the DLL) for the mouse (and the gamepad)
 *   sim_pad=0|1         warp by the gamepad's right stick as well (fit_pad.py fits the rest):
 *   sim_pad_yaw=<11 values>    the camera's turn rate about y (rad/s) at stick 0, 0.1 .. 1 to the
 *   sim_pad_pitch=<11 values>  right / about x at 0 .. 1 up (linear between; signed as fitted)
 *   sim_pad_radial=0|1  the rate by the stick's distance from the centre (1) or by each axis (0)
 *   sim_pad_tau_ms=<ms> sim_pad_lag_ms=<ms>   the stick's delay to the present, the rate's lag
 *   sim_rec_frames=<n>  sim_rec_dir=<folder>
 * Keys, with the game in front: F6 the next mode, F7 the next delay (25 50 75 100 150 ms), F8 blind
 * (F6 then picks at random and the mark is hidden), End a recording (quarter-size luma, times and the
 * mouse, for calibration), Home a snapshot (the frame and its depth).
 * Log: cloud-sim-vk.log beside the DLL, or CLOUD_SIM_VK_LOG. */
#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include "mouse_shm.h"
#include "sim_spv.h"

#define H64(x) ((uint64_t)(uintptr_t)(x))
#define MAX_SLOTS 48
#define LEVELS 12
#define MAX_IMAGES 16
#define PROF_MAX 48                      /* sim_profile: timestamps a frame, after each dispatch */
#define PROF_OCC 12                      /* ... kept apart by their order among their kernel's */
#define ENTS 16
#define KEEP_LAYOUT ((VkImageLayout)0x7FFFFFFF)
#define STAT_WORDS (64 * 40)
#define PAD_KNOTS 11                     /* the stick's rate tables: at 0, 0.1 .. 1 */

enum { K_BWARP, K_PDOWN, K_PFILL, K_PUP, K_ZOOM, K_OVERLAY, K_LUMA, K_DSTAT, K_HALF, K_CLEAR, K_SPLATZ, K_SPLATC, K_HFILL,
       K_COUNT };
static const uint32_t *const spv_code[K_COUNT] = { spv_bwarp, spv_pdown, spv_pfill, spv_pup, spv_zoom, spv_overlay, spv_luma,
						    spv_dstat, spv_half, spv_clear, spv_splatz, spv_splatc, spv_hfill };
static const size_t spv_size[K_COUNT] = { sizeof spv_bwarp, sizeof spv_pdown, sizeof spv_pfill, sizeof spv_pup,
					   sizeof spv_zoom, sizeof spv_overlay, sizeof spv_luma, sizeof spv_dstat,
					   sizeof spv_half, sizeof spv_clear, sizeof spv_splatz, sizeof spv_splatc, sizeof spv_hfill };
static const char *const kernel_name[K_COUNT] = { "bwarp", "pdown", "pfill", "pup", "zoom", "overlay", "luma", "dstat", "half",
						   "clear", "splatz", "splatc", "hfill" };
/* the keys a first-person game moves with, as bits in recordings and in the move model */
enum { KEY_W = 1, KEY_A = 2, KEY_S = 4, KEY_D = 8, KEY_SHIFT = 16, KEY_SPACE = 32, KEY_CTRL = 64 };
enum { SIM_LOCAL, SIM_CLOUD, SIM_WARP };
static const char *const mode_name[3] = { "local", "cloud", "warp" };
static const float latencies[] = { 25.0f, 50.0f, 75.0f, 100.0f, 150.0f };

/* ---------------------------------------------------------------- log */

static wchar_t here[MAX_PATH];
static FILE *logfile;
static SRWLOCK log_lock = SRWLOCK_INIT;
static LARGE_INTEGER qpf, qp0;

static void say(const char *fmt, ...)
{
	AcquireSRWLockExclusive(&log_lock);
	if (!logfile) {
		wchar_t path[MAX_PATH + 32];
		DWORD n = GetEnvironmentVariableW(L"CLOUD_SIM_VK_LOG", path, MAX_PATH);
		if (!n || n >= MAX_PATH) _snwprintf(path, MAX_PATH + 32, L"%ls\\cloud-sim-vk.log", here);
		path[MAX_PATH + 31] = 0;
		logfile = _wfopen(path, L"w");
	}
	if (logfile) {
		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		fprintf(logfile, "%9.3f ", (double)(now.QuadPart - qp0.QuadPart) / (double)qpf.QuadPart);
		va_list ap;
		va_start(ap, fmt);
		vfprintf(logfile, fmt, ap);
		va_end(ap);
		fflush(logfile);
	}
	ReleaseSRWLockExclusive(&log_lock);
}

/* ---------------------------------------------------------------- settings */

static struct {
	int on, mode, logger, slots, depth, reversed, rec_frames, blind, demo_frames, move, history, profile, pad, pad_radial;
	float latency_ms, keep, tau_ms, lag_ms, hfov, hole, cross, weapon, move_speed, move_tau_ms, move_lag_ms, jitter_ms, loss;
	float cx[3], cy[3];
	float pad_tau_ms, pad_lag_ms, pad_yaw[PAD_KNOTS], pad_pitch[PAD_KNOTS];
	float lock[4][4];
	wchar_t rec_dir[MAX_PATH], path[MAX_PATH + 32];
	FILETIME stamp;
	int loaded;
	/* the file's last values of the keys that act on change */
	char f_mode[32], f_latency[32], f_record[32], f_snapshot[32], f_dump[32], f_demo[32];
	int want_record, want_snapshot, want_dump, want_demo;
} cfg = { .on = 1, .mode = SIM_LOCAL, .logger = 1, .slots = 24, .depth = 1, .reversed = -1, .rec_frames = 600, .demo_frames = 450,
	  .move_tau_ms = -1.0f, .history = 1, .pad_radial = 1, .pad_tau_ms = 30.0f,
	  .latency_ms = 50.0f, .keep = 0.92f, .tau_ms = 30.0f, .lag_ms = 0.0f, .hfov = 90.0f, .hole = 0.02f,
	  .cross = 0.012f, .cx = { 0.0f, -5e-4f, 0.0f }, .cy = { 5e-4f, 0.0f, 0.0f } };

static char *trim(char *s)
{
	while (*s == ' ' || *s == '\t') s++;
	size_t n = strlen(s);
	while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r' || s[n - 1] == '\n')) s[--n] = 0;
	return s;
}

static int changed(char last[32], const char *v)
{
	int c = strncmp(last, v, 31) != 0;
	strncpy(last, v, 31);
	last[31] = 0;
	return c;
}

/* up to n numbers from v into out; the rest stay 0 */
static void floats(const char *v, float *out, int n)
{
	char *end;
	for (int i = 0; i < n; i++) {
		out[i] = 0.0f;
		float x = strtof(v, &end);
		if (end != v) out[i] = x, v = end;
	}
}

static void setting(const char *k, const char *v)
{
	if (!strcmp(k, "sim")) cfg.on = atoi(v) != 0;
	else if (!strcmp(k, "sim_pad")) cfg.pad = atoi(v) != 0;
	else if (!strcmp(k, "sim_pad_radial")) cfg.pad_radial = atoi(v) != 0;
	else if (!strcmp(k, "sim_pad_tau_ms")) cfg.pad_tau_ms = (float)atof(v);
	else if (!strcmp(k, "sim_pad_lag_ms")) cfg.pad_lag_ms = (float)atof(v);
	else if (!strcmp(k, "sim_pad_yaw")) floats(v, cfg.pad_yaw, PAD_KNOTS);
	else if (!strcmp(k, "sim_pad_pitch")) floats(v, cfg.pad_pitch, PAD_KNOTS);
	else if (!strcmp(k, "sim_mode")) { if (changed(cfg.f_mode, v)) cfg.mode = abs(atoi(v)) % 3; }
	else if (!strcmp(k, "sim_latency")) { if (changed(cfg.f_latency, v)) cfg.latency_ms = (float)atof(v); }
	else if (!strcmp(k, "sim_record_now")) { if (changed(cfg.f_record, v) && cfg.loaded && atoi(v) > 0) cfg.want_record = 1; }
	else if (!strcmp(k, "sim_snapshot_now")) { if (changed(cfg.f_snapshot, v) && cfg.loaded && atoi(v) > 0) cfg.want_snapshot = 1; }
	else if (!strcmp(k, "sim_dump_now")) { if (changed(cfg.f_dump, v) && cfg.loaded && atoi(v) > 0) cfg.want_dump = 1; }
	else if (!strcmp(k, "sim_demo_now")) { if (changed(cfg.f_demo, v) && cfg.loaded && atoi(v) > 0) cfg.want_demo = 1; }
	else if (!strcmp(k, "sim_demo_frames")) cfg.demo_frames = atoi(v);
	else if (!strcmp(k, "sim_move")) cfg.move = atoi(v) != 0;
	else if (!strcmp(k, "sim_move_speed")) cfg.move_speed = (float)atof(v);
	else if (!strcmp(k, "sim_move_tau_ms")) cfg.move_tau_ms = (float)atof(v);
	else if (!strcmp(k, "sim_move_lag_ms")) cfg.move_lag_ms = (float)atof(v);
	else if (!strcmp(k, "sim_jitter_ms")) cfg.jitter_ms = (float)atof(v);
	else if (!strcmp(k, "sim_history")) cfg.history = atoi(v) != 0;
	else if (!strcmp(k, "sim_profile")) cfg.profile = atoi(v) != 0;
	else if (!strcmp(k, "sim_loss")) cfg.loss = (float)atof(v);
	else if (!strcmp(k, "sim_keep")) cfg.keep = (float)atof(v);
	else if (!strcmp(k, "sim_slots")) { if (!cfg.loaded) cfg.slots = atoi(v); }
	else if (!strcmp(k, "sim_depth")) { if (!cfg.loaded) cfg.depth = atoi(v) != 0; }
	else if (!strcmp(k, "sim_depth_reversed")) cfg.reversed = !strcmp(v, "auto") ? -1 : atoi(v) != 0;
	else if (!strcmp(k, "sim_weapon_depth")) cfg.weapon = (float)atof(v);
	else if (!strcmp(k, "sim_tau_ms")) cfg.tau_ms = (float)atof(v);
	else if (!strcmp(k, "sim_lag_ms")) cfg.lag_ms = (float)atof(v);
	else if (!strcmp(k, "sim_hfov")) cfg.hfov = (float)atof(v);
	else if (!strcmp(k, "sim_hole")) cfg.hole = (float)atof(v);
	else if (!strcmp(k, "sim_cross")) cfg.cross = (float)atof(v);
	else if (!strcmp(k, "sim_cx")) sscanf(v, "%f %f %f", &cfg.cx[0], &cfg.cx[1], &cfg.cx[2]);
	else if (!strcmp(k, "sim_cy")) sscanf(v, "%f %f %f", &cfg.cy[0], &cfg.cy[1], &cfg.cy[2]);
	else if (!strncmp(k, "sim_lock", 8) && k[8] >= '1' && k[8] <= '4' && !k[9]) {
		float *r = cfg.lock[k[8] - '1'];
		r[0] = r[1] = r[2] = r[3] = 0.0f;
		sscanf(v, "%f %f %f %f", &r[0], &r[1], &r[2], &r[3]);
	} else if (!strcmp(k, "sim_logger")) cfg.logger = atoi(v) != 0;
	else if (!strcmp(k, "sim_rec_frames")) cfg.rec_frames = atoi(v);
	else if (!strcmp(k, "sim_rec_dir")) MultiByteToWideChar(CP_UTF8, 0, v, -1, cfg.rec_dir, MAX_PATH);
	else say("settings: unknown key %s\n", k);
}

/* Reads the file when it changed (always the first time); 1 when it was read. */
static int load_settings(void)
{
	if (!cfg.path[0]) {
		DWORD n = GetEnvironmentVariableW(L"CLOUD_SIM_VK_SETTINGS", cfg.path, MAX_PATH);
		if (!n || n >= MAX_PATH) _snwprintf(cfg.path, MAX_PATH + 32, L"%ls\\cloud-sim-vk.txt", here);
		cfg.path[MAX_PATH + 31] = 0;
		wchar_t profile[MAX_PATH];
		n = GetEnvironmentVariableW(L"USERPROFILE", profile, MAX_PATH);
		if (n && n < MAX_PATH - 40) _snwprintf(cfg.rec_dir, MAX_PATH, L"%ls\\datasets\\doom-rec", profile);
	}
	WIN32_FILE_ATTRIBUTE_DATA a;
	if (!GetFileAttributesExW(cfg.path, GetFileExInfoStandard, &a)) {
		if (!cfg.loaded) say("settings: no %ls; the defaults\n", cfg.path);
		cfg.loaded = 1;
		return 0;
	}
	if (cfg.loaded && CompareFileTime(&a.ftLastWriteTime, &cfg.stamp) == 0) return 0;
	cfg.stamp = a.ftLastWriteTime;
	FILE *f = _wfopen(cfg.path, L"rb");
	if (!f) return 0;
	char line[512];
	while (fgets(line, sizeof line, f)) {
		char *s = trim(line);
		if (!*s || *s == '#') continue;
		char *eq = strchr(s, '=');
		if (!eq) continue;
		*eq = 0;
		setting(trim(s), trim(eq + 1));
	}
	fclose(f);
	if (!(cfg.keep > 0.3f) || cfg.keep > 1.0f) cfg.keep = 1.0f;
	if (cfg.slots < 4) cfg.slots = 4;
	if (cfg.slots > MAX_SLOTS) cfg.slots = MAX_SLOTS;
	if (cfg.rec_frames < 1) cfg.rec_frames = 1;
	if (cfg.rec_frames > 6000) cfg.rec_frames = 6000;
	if (!(cfg.hfov > 10.0f && cfg.hfov < 170.0f)) cfg.hfov = 90.0f;
	if (!(cfg.latency_ms >= 0.0f)) cfg.latency_ms = 50.0f;
	if (cfg.demo_frames < 1) cfg.demo_frames = 1;
	if (cfg.demo_frames > 20000) cfg.demo_frames = 20000;
	char locks[256];
	int at = 0;
	locks[0] = 0;
	for (int i = 0; i < 4; i++)
		if (cfg.lock[i][2] > cfg.lock[i][0] && cfg.lock[i][3] > cfg.lock[i][1])
			at += snprintf(locks + at, sizeof locks - (size_t)at, " [%.3f %.3f %.3f %.3f]", cfg.lock[i][0], cfg.lock[i][1],
				       cfg.lock[i][2], cfg.lock[i][3]);
	say("settings%s: sim %d, mode %s, delay %.0f ms, keep %.3f, tau %.1f ms, lag %.1f ms, hfov %.2f, "
	    "cx %g %g %g, cy %g %g %g, weapon %g, reversed %d, hole %.3f, cross %.3f, depth %d, slots %d, locks%s; "
	    "move %d (speed %g near-plane units a second, tau %g ms, lag %g ms); network jitter %g ms, loss %g\n",
	    cfg.loaded ? " (again)" : "", cfg.on, mode_name[cfg.mode], cfg.latency_ms, cfg.keep, cfg.tau_ms, cfg.lag_ms,
	    cfg.hfov, cfg.cx[0], cfg.cx[1], cfg.cx[2], cfg.cy[0], cfg.cy[1], cfg.cy[2], cfg.weapon, cfg.reversed,
	    cfg.hole, cfg.cross, cfg.depth, cfg.slots, at ? locks : " none", cfg.move, cfg.move_speed,
	    cfg.move_tau_ms >= 0 ? cfg.move_tau_ms : cfg.tau_ms, cfg.move_lag_ms, cfg.jitter_ms, cfg.loss);
	if (cfg.pad)
		say("settings: the pad's right stick turns the warp too (%s; tau %.1f ms, lag %.1f ms; yaw rad/s at 0.5 %.3f, "
		    "at 1 %.3f; pitch at 0.5 %.3f, at 1 %.3f)\n", cfg.pad_radial ? "radial" : "per axis", cfg.pad_tau_ms, cfg.pad_lag_ms,
		    cfg.pad_yaw[PAD_KNOTS / 2], cfg.pad_yaw[PAD_KNOTS - 1], cfg.pad_pitch[PAD_KNOTS / 2], cfg.pad_pitch[PAD_KNOTS - 1]);
	cfg.loaded = 1;
	return 1;
}

/* ---------------------------------------------------------------- a map, 64-bit key to pointer */

typedef struct { uint64_t *k; void **v; uint8_t *st; size_t cap, n, dead; } map_t;

static size_t mix64(uint64_t x)
{
	x ^= x >> 33;
	x *= 0xff51afd7ed558ccdULL;
	x ^= x >> 33;
	x *= 0xc4ceb9fe1a85ec53ULL;
	x ^= x >> 33;
	return (size_t)x;
}

static void *map_get(const map_t *m, uint64_t key)
{
	if (!m->cap) return NULL;
	for (size_t i = mix64(key) & (m->cap - 1);; i = (i + 1) & (m->cap - 1)) {
		if (m->st[i] == 0) return NULL;
		if (m->st[i] == 1 && m->k[i] == key) return m->v[i];
	}
}

static void map_put(map_t *m, uint64_t key, void *val);

static void map_rehash(map_t *m, size_t cap)
{
	map_t n = { calloc(cap, 8), calloc(cap, sizeof(void *)), calloc(cap, 1), cap, 0, 0 };
	if (!n.k || !n.v || !n.st) {
		free(n.k), free(n.v), free(n.st);
		return;
	}
	for (size_t i = 0; i < m->cap; i++)
		if (m->st[i] == 1) map_put(&n, m->k[i], m->v[i]);
	free(m->k), free(m->v), free(m->st);
	*m = n;
}

static void map_put(map_t *m, uint64_t key, void *val)
{
	if ((m->n + m->dead + 1) * 2 > m->cap) map_rehash(m, m->cap < 64 ? 64 : (m->n * 4 > m->cap ? m->cap * 2 : m->cap));
	if (!m->cap) return;
	size_t free_at = (size_t)-1;
	for (size_t i = mix64(key) & (m->cap - 1);; i = (i + 1) & (m->cap - 1)) {
		if (m->st[i] == 1 && m->k[i] == key) {
			m->v[i] = val;
			return;
		}
		if (m->st[i] == 2 && free_at == (size_t)-1) free_at = i;
		if (m->st[i] == 0) {
			if (free_at == (size_t)-1) free_at = i;
			else m->dead--;
			m->k[free_at] = key, m->v[free_at] = val, m->st[free_at] = 1, m->n++;
			return;
		}
	}
}

static void *map_del(map_t *m, uint64_t key)
{
	if (!m->cap) return NULL;
	for (size_t i = mix64(key) & (m->cap - 1);; i = (i + 1) & (m->cap - 1)) {
		if (m->st[i] == 0) return NULL;
		if (m->st[i] == 1 && m->k[i] == key) {
			m->st[i] = 2, m->n--, m->dead++;
			return m->v[i];
		}
	}
}

/* ---------------------------------------------------------------- objects */

typedef struct inst {
	VkInstance handle;
	void *key;
	PFN_vkGetInstanceProcAddr gipa;
	PFN_vkDestroyInstance DestroyInstance;
	PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
	PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
	PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
	PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR GetPhysicalDeviceSurfaceCapabilitiesKHR;
} inst_t;

#define DEVICE_FUNCS(X)                                                                                          \
	X(DestroyDevice) X(GetDeviceQueue) X(CreateSwapchainKHR) X(DestroySwapchainKHR) X(GetSwapchainImagesKHR)    \
	X(QueuePresentKHR) X(QueueSubmit) X(CreateImage) X(DestroyImage) X(CreateImageView) X(DestroyImageView)    \
	X(CreateFramebuffer) X(DestroyFramebuffer) X(CreateRenderPass) X(DestroyRenderPass) X(CmdBeginRenderPass)  \
	X(CmdEndRenderPass) X(CmdDraw) X(CmdDrawIndexed) X(CmdDrawIndirect) X(CmdDrawIndexedIndirect)              \
	X(CmdPipelineBarrier) X(BeginCommandBuffer) X(CmdExecuteCommands) X(CreateCommandPool)                      \
	X(DestroyCommandPool) X(AllocateCommandBuffers) X(EndCommandBuffer) X(CreateFence) X(DestroyFence)          \
	X(WaitForFences) X(ResetFences) X(CreateSemaphore) X(DestroySemaphore) X(CreateBuffer) X(DestroyBuffer)    \
	X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) X(MapMemory)            \
	X(UnmapMemory) X(CreateShaderModule) X(DestroyShaderModule) X(CreateComputePipelines) X(DestroyPipeline)   \
	X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateDescriptorSetLayout)                             \
	X(DestroyDescriptorSetLayout) X(CreateDescriptorPool) X(DestroyDescriptorPool) X(AllocateDescriptorSets)   \
	X(UpdateDescriptorSets) X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants) X(CmdDispatch)    \
	X(CmdCopyImageToBuffer) X(CmdCopyBufferToImage) X(CmdCopyBuffer) X(CmdUpdateBuffer) X(CreateQueryPool)        \
	X(DestroyQueryPool) X(CmdResetQueryPool) X(CmdWriteTimestamp) X(GetQueryPoolResults) X(GetDeviceQueue2)

typedef struct img {
	VkImage image;
	VkFormat format;
	VkExtent3D extent;
	VkImageUsageFlags usage;
	VkImageAspectFlags aspect;
	VkImageLayout layout;            /* after the last submission, by its render passes and barriers */
	uint64_t draws, total;           /* this frame's draws, all draws */
	VkAttachmentStoreOp store;       /* the depth store op of the last render pass seen with it */
	VkSharingMode sharing;           /* as made: exclusive (one family) or concurrent */
	int dead;
	struct img *next_all;
} img_t;

typedef struct { img_t *depth; uint32_t index; } fb_t;
typedef struct { uint32_t n; VkImageLayout final[16]; VkAttachmentStoreOp store[16]; } rp_t;
typedef struct { img_t *img; uint32_t draws; VkImageLayout layout; } ent_t;
typedef struct { int n, cur; uint32_t loose; ent_t e[ENTS]; } cmd_t;

typedef struct sc {
	VkSwapchainKHR handle;
	VkFormat format;
	VkExtent2D extent;
	uint32_t count;
	VkImage images[MAX_IMAGES];
	int transfer;                    /* TRANSFER_SRC | TRANSFER_DST added */
} sc_t;

typedef struct buf { VkBuffer b; VkDeviceMemory m; VkDeviceSize size; void *map; } buf_t;

struct params {                          /* sim.comp's Params, std430 */
	float C[3][4];
	float f, cx, cy, keep;
	uint32_t W, H, RW, RH;
	uint32_t mark, dflags, lockz, bgra;
	uint32_t hole, cross, nlock, depth_ok;
	uint32_t lock[4][4];
	float T[4];                      /* the camera's move, old frame's axes, near-plane units */
	float HC[9][4];                  /* older frames for the holes: R'^T by rows, three frames */
};
#define PARAMS_BYTES 1024                /* the params buffer; vkCmdUpdateBuffer writes the struct into it */
_Static_assert(sizeof(struct params) <= PARAMS_BYTES && sizeof(struct params) % 4 == 0, "params outgrew its buffer");

/* why a present passed the simulator by (the log says so: F6, End and the rest do nothing then) */
enum { SKIP_NONE, SKIP_IMAGE, SKIP_FAMILY, SKIP_QUEUE, SKIP_CMDBUF, SKIP_SWAPCHAIN, SKIP_REFUSED, SKIP_FAILED, SKIP_MANY,
       SKIP_OTHER, SKIP_N };
static const char *const skip_name[SKIP_N] = {
	"-", "an image index past the swapchain's", "a queue of another family", "a queue the layer never saw got",
	"our command buffer would not begin or end", "a swapchain the layer did not see made", "a swapchain the simulator refused",
	"the simulator stopped after an error", "more than one swapchain at once", "another reason" };

typedef struct sim {
	int pipes, ready, failed, skip, graphics;
	sc_t *sc, *refused;
	VkQueue queue, skip_queue;
	uint32_t family, W, H, nimg, bgra, slots, levels, skip_fam;
	ULONGLONG move_ok;               /* the simulator follows the game to another queue family at most this often */
	VkDescriptorSetLayout dsl;
	VkPipelineLayout pl;
	VkPipeline pipe[K_COUNT];
	VkCommandPool pool;
	VkCommandBuffer cmd[MAX_IMAGES];
	VkFence fence[MAX_IMAGES];
	VkSemaphore sem[MAX_IMAGES];
	int used[MAX_IMAGES], rec_at[MAX_IMAGES], stat_at[MAX_IMAGES], snap_at[MAX_IMAGES];
	LONGLONG rec_q[MAX_IMAGES];
	int rec_m[MAX_IMAGES];
	uint32_t stat_fl[MAX_IMAGES];
	buf_t color[MAX_SLOTS], depth[MAX_SLOTS], zbuf, out, fin, pkeys, pcolor, params, rec, snap;
	VkDeviceSize depth_cap, koff[LEVELS + 1], coff[LEVELS + 1];
	uint32_t rec_stride, luma_words, wpr, lumw, lumh;
	uint32_t lw[LEVELS + 1], lh[LEVELS + 1];
	VkDescriptorPool dpool;
	VkDescriptorSet set_frame[MAX_SLOTS], set_level[LEVELS], set_zoom;
	LONGLONG qpc[MAX_SLOTS], arrive[MAX_SLOTS];   /* made, and arrives at the cloud player */
	uint32_t drw[MAX_SLOTS], drh[MAX_SLOTS], dfl[MAX_SLOTS];
	int dok[MAX_SLOTS], lost[MAX_SLOTS];
	uint64_t rng;
	double jit;
	unsigned long long lost_count;
	unsigned long long frames;
	img_t *dimg;
	HANDLE mapping, dump_event;
	struct mouse_shm *shm;
	int key[8], short_ring_said, trial;
	ULONGLONG next_poll, next_log;
	unsigned long long log_frames, next_stat;
	LONGLONG log_q;
	int64_t log_mice, log_pads;
	int log_pad_on;
	int rec_left, rec_total, rec_got;
	unsigned char *rec_data;
	LONGLONG *rec_qpc;
	int *rec_mode;
	wchar_t rec_path[MAX_PATH];
	int snap_busy, snap_kind;
	uint32_t snap_bytes0, snap_bytes1;
	char snap_info[2048];
	int reversed, rev_against;
	int dpending;                    /* frames another depth image has led the one followed */
	/* our own GPU time a frame: a timestamp at the top and at the bottom of our command buffer */
	VkQueryPool qpool;
	int ts_ok, ts_at[MAX_IMAGES];
	uint32_t ts_bits;
	double ts_sum_ms, ts_max_ms, ts_part[3];
	unsigned ts_n;
	/* sim_profile: a timestamp after each dispatch too, the time by kernel and by its order in the frame */
	uint32_t ts_stride;              /* queries an image: 4, then PROF_MAX */
	int prof_img, prof_n[MAX_IMAGES];
	unsigned char prof_k[MAX_IMAGES][PROF_MAX];
	double prof_ms[K_COUNT][PROF_OCC + 1];
	unsigned prof_cnt[K_COUNT], prof_frames;
	/* the keys a game moves with, sampled at each present: per ring slot, and into the recordings */
	uint32_t keys[MAX_SLOTS], rec_kb[MAX_IMAGES], demo_kb[MAX_IMAGES];
	int *rec_keys;
	HANDLE keys_map;                 /* a test's stand-in for the keyboard (Local\cloud-sim-vk-keys-<pid>) */
	volatile LONG *keys_shm;
	double last_T[3];
	/* the demo recording: half-size colour + nearness per frame, streamed to disk */
	uint32_t demo_off, demo_doff, demo_cwords, demo_dwords;   /* in each image's rec region, words */
	int demo_ok, demo_left, demo_total, demo_got, demo_at[MAX_IMAGES], demo_m[MAX_IMAGES];
	LONGLONG demo_q[MAX_IMAGES];
	uint32_t demo_fl[MAX_IMAGES];
	/* A present family without graphics (DOOM presents from its compute queue; Vulkan copies depth only on
	 * queues with graphics): the game's depth is copied on its own graphics queue into gstage[i] (gcopy),
	 * and our work on the present queue waits for that (gsem[i]) and takes it into the ring, so the ring
	 * is still written on the present queue only */
	VkQueue gqueue;
	uint32_t gfamily;
	VkCommandPool gpool;
	VkCommandBuffer gcmd[MAX_IMAGES];
	VkFence gfence[MAX_IMAGES];
	VkSemaphore gsem[MAX_IMAGES];
	buf_t gstage[MAX_IMAGES];
	VkDeviceSize gbytes[MAX_IMAGES];
	int gused[MAX_IMAGES], gsig[MAX_IMAGES];   /* a copy submitted (its fence to wait); gsem signalled, not yet waited */
} sim_t;

typedef struct device {
	VkDevice handle;
	void *key;
	VkPhysicalDevice phys;
	inst_t *inst;
	PFN_vkGetDeviceProcAddr gdpa;
	PFN_vkSetDeviceLoaderData set_loader_data;
#define X(n) PFN_vk##n n;
	DEVICE_FUNCS(X)
#undef X
	VkPhysicalDeviceMemoryProperties mem;
	VkPhysicalDeviceProperties props;
	sim_t sim;
} device_t;

static SRWLOCK obj_lock = SRWLOCK_INIT;      /* the maps below but cmds */
static SRWLOCK cmd_lock = SRWLOCK_INIT;      /* cmds */
static SRWLOCK sub_lock = SRWLOCK_INIT;      /* img_t layout and draws */
/* Queue submits, the game's and ours: the layer submits to the game's graphics queue too (gcopy), and the
 * submits to one queue must not overlap; taken before sub_lock */
static SRWLOCK q_lock = SRWLOCK_INIT;
static map_t insts, devs, queues, images, views, fbs, rps, swapchains, cmds;
static device_t *only_dev;
static img_t *all_imgs;                      /* every img_t ever made: freed with nothing */
static int ndevs;

static void *dkey_of(const void *h) { return *(void *const *)h; }

static inst_t *inst_of(const void *h)
{
	AcquireSRWLockShared(&obj_lock);
	inst_t *in = map_get(&insts, H64(dkey_of(h)));
	ReleaseSRWLockShared(&obj_lock);
	return in;
}

static device_t *dev_of(const void *h)
{
	void *k = dkey_of(h);
	device_t *d = only_dev;
	if (d && d->key == k) return d;
	AcquireSRWLockShared(&obj_lock);
	d = map_get(&devs, H64(k));
	ReleaseSRWLockShared(&obj_lock);
	return d;
}

static __declspec(thread) VkCommandBuffer tl_cb;
static __declspec(thread) cmd_t *tl_cmd;

static cmd_t *cmd_of(VkCommandBuffer cb)
{
	if (tl_cb == cb && tl_cmd) return tl_cmd;
	AcquireSRWLockShared(&cmd_lock);
	cmd_t *c = map_get(&cmds, H64(cb));
	ReleaseSRWLockShared(&cmd_lock);
	if (!c) {
		cmd_t *n = calloc(1, sizeof *n);
		if (!n) return NULL;
		n->cur = -1;
		AcquireSRWLockExclusive(&cmd_lock);
		c = map_get(&cmds, H64(cb));
		if (!c) {
			map_put(&cmds, H64(cb), n);
			c = map_get(&cmds, H64(cb));
		}
		ReleaseSRWLockExclusive(&cmd_lock);
		if (c != n) free(n);
		if (!c) return NULL;
	}
	tl_cb = cb, tl_cmd = c;
	return c;
}

static int ent_of(cmd_t *c, img_t *img)
{
	for (int i = 0; i < c->n; i++)
		if (c->e[i].img == img) return i;
	if (c->n >= ENTS) return -1;
	c->e[c->n] = (ent_t){ img, 0, KEEP_LAYOUT };
	return c->n++;
}

static int is_depth(VkFormat f)
{
	switch (f) {
	case VK_FORMAT_D16_UNORM: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D32_SFLOAT:
	case VK_FORMAT_D16_UNORM_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
		return 1;
	default:
		return 0;
	}
}

static VkImageAspectFlags aspects_of(VkFormat f)
{
	return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT
		       ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
		       : VK_IMAGE_ASPECT_DEPTH_BIT;
}

/* 1 24-bit in 32, 4 16-bit, 0 float */
static uint32_t depth_kind(VkFormat f)
{
	if (f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D16_UNORM_S8_UINT) return 4;
	if (f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D24_UNORM_S8_UINT) return 1;
	return 0;
}

/* ---------------------------------------------------------------- instance */

static VKAPI_ATTR VkResult VKAPI_CALL h_CreateInstance(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *a,
							VkInstance *out)
{
	VkLayerInstanceCreateInfo *link = (VkLayerInstanceCreateInfo *)ci->pNext;
	while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && link->function == VK_LAYER_LINK_INFO))
		link = (VkLayerInstanceCreateInfo *)link->pNext;
	if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
	PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
	link->u.pLayerInfo = link->u.pLayerInfo->pNext;
	PFN_vkCreateInstance create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
	if (!create) return VK_ERROR_INITIALIZATION_FAILED;
	VkResult r = create(ci, a, out);
	if (r != VK_SUCCESS) return r;
	if (!cfg.loaded) {
		wchar_t exe[MAX_PATH];
		GetModuleFileNameW(NULL, exe, MAX_PATH);
		say("cloud_sim_vk in %ls (pid %lu)\n", exe, GetCurrentProcessId());
		load_settings();
	}
	inst_t *in = calloc(1, sizeof *in);
	if (!in) return r;
	in->handle = *out, in->key = dkey_of(*out), in->gipa = gipa;
	in->DestroyInstance = (PFN_vkDestroyInstance)gipa(*out, "vkDestroyInstance");
	in->GetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)gipa(*out, "vkGetPhysicalDeviceMemoryProperties");
	in->GetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)gipa(*out, "vkGetPhysicalDeviceProperties");
	in->GetPhysicalDeviceQueueFamilyProperties =
		(PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(*out, "vkGetPhysicalDeviceQueueFamilyProperties");
	in->GetPhysicalDeviceSurfaceCapabilitiesKHR =
		(PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)gipa(*out, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
	AcquireSRWLockExclusive(&obj_lock);
	map_put(&insts, H64(in->key), in);
	ReleaseSRWLockExclusive(&obj_lock);
	const VkApplicationInfo *ai = ci->pApplicationInfo;
	say("instance: api %u.%u, app %s, engine %s, %u extensions\n", ai ? VK_VERSION_MAJOR(ai->apiVersion) : 1,
	    ai ? VK_VERSION_MINOR(ai->apiVersion) : 0, ai && ai->pApplicationName ? ai->pApplicationName : "-",
	    ai && ai->pEngineName ? ai->pEngineName : "-", ci->enabledExtensionCount);
	return r;
}

static VKAPI_ATTR void VKAPI_CALL h_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *a)
{
	if (!instance) return;
	void *k = dkey_of(instance);
	AcquireSRWLockExclusive(&obj_lock);
	inst_t *in = map_del(&insts, H64(k));
	ReleaseSRWLockExclusive(&obj_lock);
	if (!in) return;
	in->DestroyInstance(instance, a);
	free(in);
}

/* ---------------------------------------------------------------- device */

static VKAPI_ATTR VkResult VKAPI_CALL h_CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo *ci,
						      const VkAllocationCallbacks *a, VkDevice *out)
{
	VkLayerDeviceCreateInfo *link = NULL, *data = NULL;
	for (VkLayerDeviceCreateInfo *c = (VkLayerDeviceCreateInfo *)ci->pNext; c; c = (VkLayerDeviceCreateInfo *)c->pNext)
		if (c->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO) {
			if (c->function == VK_LAYER_LINK_INFO && !link) link = c;
			else if (c->function == VK_LOADER_DATA_CALLBACK && !data) data = c;
		}
	inst_t *in = inst_of(phys);
	if (!link || !link->u.pLayerInfo || !in) return VK_ERROR_INITIALIZATION_FAILED;
	PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
	PFN_vkGetDeviceProcAddr gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
	link->u.pLayerInfo = link->u.pLayerInfo->pNext;
	PFN_vkCreateDevice create = (PFN_vkCreateDevice)gipa(in->handle, "vkCreateDevice");
	if (!create) return VK_ERROR_INITIALIZATION_FAILED;
	VkResult r = create(phys, ci, a, out);
	if (r != VK_SUCCESS) return r;
	device_t *d = calloc(1, sizeof *d);
	if (!d) return r;
	d->handle = *out, d->key = dkey_of(*out), d->phys = phys, d->inst = in, d->gdpa = gdpa;
	d->set_loader_data = data ? data->u.pfnSetDeviceLoaderData : NULL;
#define X(n) d->n = (PFN_vk##n)gdpa(*out, "vk" #n);
	DEVICE_FUNCS(X)
#undef X
	in->GetPhysicalDeviceMemoryProperties(phys, &d->mem);
	in->GetPhysicalDeviceProperties(phys, &d->props);
	AcquireSRWLockExclusive(&obj_lock);
	map_put(&devs, H64(d->key), d);
	ndevs++;
	only_dev = ndevs == 1 ? d : NULL;
	ReleaseSRWLockExclusive(&obj_lock);
	char fams[160];
	int fat = 0;
	fams[0] = 0;
	for (uint32_t q = 0; q < ci->queueCreateInfoCount && q < 8 && fat < (int)sizeof fams - 24; q++)
		fat += snprintf(fams + fat, sizeof fams - (size_t)fat, "%s%u x%u", q ? ", " : "", ci->pQueueCreateInfos[q].queueFamilyIndex,
				ci->pQueueCreateInfos[q].queueCount);
	say("device: %s, api %u.%u.%u, driver 0x%X, %u queue families asked (family x queues: %s), %u extensions; depth following %s\n",
	    d->props.deviceName, VK_VERSION_MAJOR(d->props.apiVersion), VK_VERSION_MINOR(d->props.apiVersion),
	    VK_VERSION_PATCH(d->props.apiVersion), d->props.driverVersion, ci->queueCreateInfoCount, fams,
	    ci->enabledExtensionCount, cfg.depth ? "on" : "off");
	return r;
}

static void sim_release(device_t *d, sim_t *s, int all);
static void demo_finish(sim_t *s);

static VKAPI_ATTR void VKAPI_CALL h_DestroyDevice(VkDevice device, const VkAllocationCallbacks *a)
{
	if (!device) return;
	device_t *d = dev_of(device);
	if (!d) return;
	sim_release(d, &d->sim, 1);
	AcquireSRWLockExclusive(&obj_lock);
	for (size_t i = 0; i < queues.cap; i++)      /* its queues go with it (graphics_queue reads through them) */
		if (queues.st[i] == 1 && dkey_of((const void *)(uintptr_t)queues.k[i]) == d->key)
			queues.st[i] = 2, queues.n--, queues.dead++;
	map_del(&devs, H64(d->key));
	ndevs--;
	only_dev = NULL;
	if (ndevs == 1)
		for (size_t i = 0; i < devs.cap; i++)
			if (devs.st[i] == 1) only_dev = devs.v[i];
	ReleaseSRWLockExclusive(&obj_lock);
	d->DestroyDevice(device, a);
	say("device destroyed\n");
	free(d);
}

static VKAPI_ATTR void VKAPI_CALL h_GetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue *q)
{
	device_t *d = dev_of(device);
	d->GetDeviceQueue(device, family, index, q);
	if (*q) {
		AcquireSRWLockExclusive(&obj_lock);
		map_put(&queues, H64(*q), (void *)(uintptr_t)(family + 1));
		ReleaseSRWLockExclusive(&obj_lock);
	}
}

/* the same for queues got the Vulkan 1.1 way (a layer above, such as an overlay, may use it) */
static VKAPI_ATTR void VKAPI_CALL h_GetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2 *info, VkQueue *q)
{
	device_t *d = dev_of(device);
	d->GetDeviceQueue2(device, info, q);
	if (*q && info) {
		AcquireSRWLockExclusive(&obj_lock);
		map_put(&queues, H64(*q), (void *)(uintptr_t)(info->queueFamilyIndex + 1));
		ReleaseSRWLockExclusive(&obj_lock);
		say("queue got through vkGetDeviceQueue2: family %u, index %u, flags 0x%X\n", info->queueFamilyIndex, info->queueIndex,
		    (unsigned)info->flags);
	}
}

/* ---------------------------------------------------------------- following the depth buffer */

static VKAPI_ATTR VkResult VKAPI_CALL h_CreateImage(VkDevice device, const VkImageCreateInfo *ci, const VkAllocationCallbacks *a,
						     VkImage *out)
{
	device_t *d = dev_of(device);
	int follow = is_depth(ci->format) && (ci->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) && ci->imageType == VK_IMAGE_TYPE_2D
		     && ci->samples == VK_SAMPLE_COUNT_1_BIT;
	if (!follow) return d->CreateImage(device, ci, a, out);
	VkImageCreateInfo c = *ci;
	c.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	VkResult r = d->CreateImage(device, &c, a, out);
	if (r != VK_SUCCESS) {
		c.usage = ci->usage;
		r = d->CreateImage(device, &c, a, out);
	}
	if (r != VK_SUCCESS) return r;
	img_t *img = calloc(1, sizeof *img);
	if (!img) return r;
	img->image = *out, img->format = ci->format, img->extent = ci->extent, img->usage = c.usage;
	img->aspect = aspects_of(ci->format), img->layout = ci->initialLayout, img->store = VK_ATTACHMENT_STORE_OP_STORE;
	img->sharing = ci->sharingMode;
	AcquireSRWLockExclusive(&obj_lock);
	map_put(&images, H64(*out), img);
	img->next_all = all_imgs;
	all_imgs = img;
	ReleaseSRWLockExclusive(&obj_lock);
	return r;
}

static VKAPI_ATTR void VKAPI_CALL h_DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks *a)
{
	device_t *d = dev_of(device);
	if (image) {
		AcquireSRWLockExclusive(&obj_lock);
		img_t *img = map_del(&images, H64(image));
		if (img) img->dead = 1;
		ReleaseSRWLockExclusive(&obj_lock);
	}
	d->DestroyImage(device, image, a);
}

static VKAPI_ATTR VkResult VKAPI_CALL h_CreateImageView(VkDevice device, const VkImageViewCreateInfo *ci,
							 const VkAllocationCallbacks *a, VkImageView *out)
{
	device_t *d = dev_of(device);
	VkResult r = d->CreateImageView(device, ci, a, out);
	if (r == VK_SUCCESS) {
		AcquireSRWLockExclusive(&obj_lock);
		img_t *img = map_get(&images, H64(ci->image));
		if (img) map_put(&views, H64(*out), img);
		ReleaseSRWLockExclusive(&obj_lock);
	}
	return r;
}

static VKAPI_ATTR void VKAPI_CALL h_DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks *a)
{
	device_t *d = dev_of(device);
	if (view) {
		AcquireSRWLockExclusive(&obj_lock);
		map_del(&views, H64(view));
		ReleaseSRWLockExclusive(&obj_lock);
	}
	d->DestroyImageView(device, view, a);
}

static VKAPI_ATTR VkResult VKAPI_CALL h_CreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo *ci,
							   const VkAllocationCallbacks *a, VkFramebuffer *out)
{
	device_t *d = dev_of(device);
	VkResult r = d->CreateFramebuffer(device, ci, a, out);
	if (r != VK_SUCCESS || (ci->flags & 1 /* IMAGELESS */) || !ci->pAttachments) return r;
	AcquireSRWLockExclusive(&obj_lock);
	img_t *depth = NULL;
	uint32_t index = 0;
	for (uint32_t i = 0; i < ci->attachmentCount && !depth; i++)
		if ((depth = map_get(&views, H64(ci->pAttachments[i])))) index = i;
	if (depth) {
		fb_t *fb = calloc(1, sizeof *fb);
		if (fb) {
			fb->depth = depth, fb->index = index;
			free(map_del(&fbs, H64(*out)));
			map_put(&fbs, H64(*out), fb);
		}
	}
	ReleaseSRWLockExclusive(&obj_lock);
	return r;
}

static VKAPI_ATTR void VKAPI_CALL h_DestroyFramebuffer(VkDevice device, VkFramebuffer fb, const VkAllocationCallbacks *a)
{
	device_t *d = dev_of(device);
	if (fb) {
		AcquireSRWLockExclusive(&obj_lock);
		void *old = map_del(&fbs, H64(fb));
		ReleaseSRWLockExclusive(&obj_lock);
		(void)old;               /* a recorded command buffer may still point at it: kept */
	}
	d->DestroyFramebuffer(device, fb, a);
}

static VKAPI_ATTR VkResult VKAPI_CALL h_CreateRenderPass(VkDevice device, const VkRenderPassCreateInfo *ci,
							  const VkAllocationCallbacks *a, VkRenderPass *out)
{
	device_t *d = dev_of(device);
	VkResult r = d->CreateRenderPass(device, ci, a, out);
	if (r != VK_SUCCESS) return r;
	rp_t *rp = calloc(1, sizeof *rp);
	if (!rp) return r;
	rp->n = ci->attachmentCount < 16 ? ci->attachmentCount : 16;
	for (uint32_t i = 0; i < rp->n; i++) {
		rp->final[i] = ci->pAttachments[i].finalLayout;
		rp->store[i] = ci->pAttachments[i].storeOp;
	}
	AcquireSRWLockExclusive(&obj_lock);
	map_put(&rps, H64(*out), rp);
	ReleaseSRWLockExclusive(&obj_lock);
	return r;
}

static VKAPI_ATTR void VKAPI_CALL h_DestroyRenderPass(VkDevice device, VkRenderPass rp, const VkAllocationCallbacks *a)
{
	device_t *d = dev_of(device);
	if (rp) {
		AcquireSRWLockExclusive(&obj_lock);
		map_del(&rps, H64(rp));      /* kept, like a framebuffer */
		ReleaseSRWLockExclusive(&obj_lock);
	}
	d->DestroyRenderPass(device, rp, a);
}

static VKAPI_ATTR VkResult VKAPI_CALL h_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi)
{
	device_t *d = dev_of(cb);
	cmd_t *c = cmd_of(cb);
	if (c) c->n = 0, c->cur = -1, c->loose = 0;
	return d->BeginCommandBuffer(cb, bi);
}

static VKAPI_ATTR void VKAPI_CALL h_CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo *bi, VkSubpassContents contents)
{
	device_t *d = dev_of(cb);
	cmd_t *c = cmd_of(cb);
	if (c) {
		AcquireSRWLockShared(&obj_lock);
		fb_t *fb = map_get(&fbs, H64(bi->framebuffer));
		rp_t *rp = map_get(&rps, H64(bi->renderPass));
		img_t *img = fb ? fb->depth : NULL;
		uint32_t at = fb ? fb->index : 0;
		VkImageLayout final = rp && at < rp->n ? rp->final[at] : KEEP_LAYOUT;
		VkAttachmentStoreOp store = rp && at < rp->n ? rp->store[at] : VK_ATTACHMENT_STORE_OP_STORE;
		ReleaseSRWLockShared(&obj_lock);
		c->cur = -1;
		if (img) {
			int i = ent_of(c, img);
			c->cur = i;
			if (i >= 0 && final != KEEP_LAYOUT) c->e[i].layout = final;
			img->store = store;
		}
	}
	d->CmdBeginRenderPass(cb, bi, contents);
}

static VKAPI_ATTR void VKAPI_CALL h_CmdEndRenderPass(VkCommandBuffer cb)
{
	device_t *d = dev_of(cb);
	cmd_t *c = cmd_of(cb);
	if (c) c->cur = -1;
	d->CmdEndRenderPass(cb);
}

static void count_draw(VkCommandBuffer cb)
{
	cmd_t *c = cmd_of(cb);
	if (!c) return;
	if (c->cur >= 0) c->e[c->cur].draws++;
	else c->loose++;
}

static VKAPI_ATTR void VKAPI_CALL h_CmdDraw(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, uint32_t e)
{
	count_draw(cb);
	dev_of(cb)->CmdDraw(cb, a, b, c, e);
}

static VKAPI_ATTR void VKAPI_CALL h_CmdDrawIndexed(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, int32_t e, uint32_t f)
{
	count_draw(cb);
	dev_of(cb)->CmdDrawIndexed(cb, a, b, c, e, f);
}

static VKAPI_ATTR void VKAPI_CALL h_CmdDrawIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s)
{
	count_draw(cb);
	dev_of(cb)->CmdDrawIndirect(cb, b, o, n, s);
}

static VKAPI_ATTR void VKAPI_CALL h_CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s)
{
	count_draw(cb);
	dev_of(cb)->CmdDrawIndexedIndirect(cb, b, o, n, s);
}

static VKAPI_ATTR void VKAPI_CALL h_CmdExecuteCommands(VkCommandBuffer cb, uint32_t n, const VkCommandBuffer *list)
{
	device_t *d = dev_of(cb);
	cmd_t *c = cmd_of(cb);
	for (uint32_t i = 0; c && i < n; i++) {
		cmd_t *s = cmd_of(list[i]);
		if (!s) continue;
		if (c->cur >= 0) c->e[c->cur].draws += s->loose;
		for (int j = 0; j < s->n; j++) {
			int at = ent_of(c, s->e[j].img);
			if (at < 0) continue;
			c->e[at].draws += s->e[j].draws;
			if (s->e[j].layout != KEEP_LAYOUT) c->e[at].layout = s->e[j].layout;
		}
		tl_cb = NULL;                    /* cmd_of(list[i]) took the cache */
		c = cmd_of(cb);
	}
	d->CmdExecuteCommands(cb, n, list);
}

static VKAPI_ATTR void VKAPI_CALL h_CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
							VkDependencyFlags dep, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb,
							const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *ib)
{
	device_t *d = dev_of(cb);
	if (ni) {
		cmd_t *c = cmd_of(cb);
		AcquireSRWLockShared(&obj_lock);
		for (uint32_t i = 0; c && i < ni; i++) {
			img_t *img = map_get(&images, H64(ib[i].image));
			if (!img) continue;
			int at = ent_of(c, img);
			if (at >= 0) c->e[at].layout = ib[i].newLayout;
		}
		ReleaseSRWLockShared(&obj_lock);
	}
	d->CmdPipelineBarrier(cb, src, dst, dep, nm, m, nb, b, ni, ib);
}

static VKAPI_ATTR VkResult VKAPI_CALL h_QueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo *si, VkFence fence)
{
	device_t *d = dev_of(queue);
	AcquireSRWLockExclusive(&q_lock);            /* the layouts below hold from here to the submit */
	AcquireSRWLockExclusive(&sub_lock);
	AcquireSRWLockShared(&cmd_lock);
	for (uint32_t i = 0; i < n; i++)
		for (uint32_t j = 0; j < si[i].commandBufferCount; j++) {
			cmd_t *c = map_get(&cmds, H64(si[i].pCommandBuffers[j]));
			if (!c) continue;
			for (int e = 0; e < c->n; e++) {
				img_t *img = c->e[e].img;
				img->draws += c->e[e].draws;
				img->total += c->e[e].draws;
				if (c->e[e].layout != KEEP_LAYOUT) img->layout = c->e[e].layout;
			}
		}
	ReleaseSRWLockShared(&cmd_lock);
	ReleaseSRWLockExclusive(&sub_lock);
	VkResult r = d->QueueSubmit(queue, n, si, fence);
	ReleaseSRWLockExclusive(&q_lock);
	return r;
}

/* ---------------------------------------------------------------- swapchains */

static VKAPI_ATTR VkResult VKAPI_CALL h_CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR *ci,
							    const VkAllocationCallbacks *a, VkSwapchainKHR *out)
{
	device_t *d = dev_of(device);
	VkSurfaceCapabilitiesKHR caps;
	memset(&caps, 0, sizeof caps);
	if (d->inst->GetPhysicalDeviceSurfaceCapabilitiesKHR)
		d->inst->GetPhysicalDeviceSurfaceCapabilitiesKHR(d->phys, ci->surface, &caps);
	const VkImageUsageFlags want = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	VkSwapchainCreateInfoKHR c = *ci;
	int transfer = (caps.supportedUsageFlags & want) == want;
	if (transfer) c.imageUsage |= want;
	VkResult r = d->CreateSwapchainKHR(device, &c, a, out);
	if (r != VK_SUCCESS && transfer) {
		transfer = 0;
		r = d->CreateSwapchainKHR(device, ci, a, out);
	}
	if (r != VK_SUCCESS) return r;
	sc_t *sc = calloc(1, sizeof *sc);
	if (!sc) return r;
	sc->handle = *out, sc->format = ci->imageFormat, sc->extent = ci->imageExtent, sc->transfer = transfer;
	uint32_t count = 0;
	d->GetSwapchainImagesKHR(device, *out, &count, NULL);
	if (count > MAX_IMAGES) count = MAX_IMAGES;
	d->GetSwapchainImagesKHR(device, *out, &count, sc->images);
	sc->count = count;
	AcquireSRWLockExclusive(&obj_lock);
	map_put(&swapchains, H64(*out), sc);
	ReleaseSRWLockExclusive(&obj_lock);
	say("swapchain: %ux%u, format %d, %u images (min %u), present mode %d, usage 0x%X, sharing %s (%u families)%s\n",
	    sc->extent.width, sc->extent.height, (int)sc->format, count, ci->minImageCount, (int)ci->presentMode,
	    (unsigned)c.imageUsage, ci->imageSharingMode == VK_SHARING_MODE_CONCURRENT ? "concurrent" : "exclusive",
	    ci->imageSharingMode == VK_SHARING_MODE_CONCURRENT ? ci->queueFamilyIndexCount : 1u,
	    transfer ? "" : " (no transfer usage: the simulator cannot run on it)");
	return r;
}

static VKAPI_ATTR void VKAPI_CALL h_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks *a)
{
	device_t *d = dev_of(device);
	if (swapchain) {
		AcquireSRWLockExclusive(&obj_lock);
		sc_t *sc = map_del(&swapchains, H64(swapchain));
		ReleaseSRWLockExclusive(&obj_lock);
		if (sc && d->sim.sc == sc) sim_release(d, &d->sim, 0);
		if (sc && d->sim.refused == sc) d->sim.refused = NULL;
		free(sc);
	}
	d->DestroySwapchainKHR(device, swapchain, a);
}

/* ---------------------------------------------------------------- the simulator's resources */

static uint32_t memtype(device_t *d, uint32_t bits, VkMemoryPropertyFlags want)
{
	for (uint32_t i = 0; i < d->mem.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (d->mem.memoryTypes[i].propertyFlags & want) == want) return i;
	return UINT32_MAX;
}

/* a buffer, shared by two queue families when f0 != f1 (concurrent: no ownership transfers) */
static int make_buf_in(device_t *d, buf_t *b, VkDeviceSize size, int host, uint32_t f0, uint32_t f1)
{
	VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bi.size = size ? size : 256;
	bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
		   | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	uint32_t fams[2] = { f0, f1 };
	bi.sharingMode = f0 != f1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
	bi.queueFamilyIndexCount = f0 != f1 ? 2u : 0u;
	bi.pQueueFamilyIndices = f0 != f1 ? fams : NULL;
	if (d->CreateBuffer(d->handle, &bi, NULL, &b->b) != VK_SUCCESS) return 0;
	VkMemoryRequirements r;
	d->GetBufferMemoryRequirements(d->handle, b->b, &r);
	uint32_t t = UINT32_MAX;
	if (host) {
		t = memtype(d, r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
							 | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
		if (t == UINT32_MAX)
			t = memtype(d, r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	} else {
		t = memtype(d, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (t == UINT32_MAX) t = memtype(d, r.memoryTypeBits, 0);
	}
	VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, r.size, t };
	if (t == UINT32_MAX || d->AllocateMemory(d->handle, &ai, NULL, &b->m) != VK_SUCCESS) {
		d->DestroyBuffer(d->handle, b->b, NULL);
		b->b = VK_NULL_HANDLE;
		return 0;
	}
	d->BindBufferMemory(d->handle, b->b, b->m, 0);
	b->size = bi.size;
	if (host && d->MapMemory(d->handle, b->m, 0, VK_WHOLE_SIZE, 0, &b->map) != VK_SUCCESS) {
		b->map = NULL;
		return 0;
	}
	return 1;
}

static int make_buf(device_t *d, buf_t *b, VkDeviceSize size, int host)
{
	return make_buf_in(d, b, size, host, 0, 0);
}

static void free_buf(device_t *d, buf_t *b)
{
	if (b->map) d->UnmapMemory(d->handle, b->m);
	if (b->b) d->DestroyBuffer(d->handle, b->b, NULL);
	if (b->m) d->FreeMemory(d->handle, b->m, NULL);
	memset(b, 0, sizeof *b);
}

static int make_pipelines(device_t *d, sim_t *s)
{
	VkDescriptorSetLayoutBinding bind[8];
	for (uint32_t i = 0; i < 8; i++)
		bind[i] = (VkDescriptorSetLayoutBinding){ i, i == 6 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
							  VK_SHADER_STAGE_COMPUTE_BIT, NULL };   /* 6: the params, constant */
	VkDescriptorSetLayoutCreateInfo li = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 8, bind };
	if (d->CreateDescriptorSetLayout(d->handle, &li, NULL, &s->dsl) != VK_SUCCESS) return 0;
	VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
	VkPipelineLayoutCreateInfo pi = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &s->dsl, 1, &pr };
	if (d->CreatePipelineLayout(d->handle, &pi, NULL, &s->pl) != VK_SUCCESS) return 0;
	for (int k = 0; k < K_COUNT; k++) {
		VkShaderModuleCreateInfo mi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, spv_size[k], spv_code[k] };
		VkShaderModule mod;
		if (d->CreateShaderModule(d->handle, &mi, NULL, &mod) != VK_SUCCESS) {
			say("sim: kernel %s: no shader module\n", kernel_name[k]);
			return 0;
		}
		VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
		ci.stage = (VkPipelineShaderStageCreateInfo){ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
							      VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", NULL };
		ci.layout = s->pl;
		VkResult r = d->CreateComputePipelines(d->handle, VK_NULL_HANDLE, 1, &ci, NULL, &s->pipe[k]);
		d->DestroyShaderModule(d->handle, mod, NULL);
		if (r != VK_SUCCESS) {
			say("sim: kernel %s: no pipeline (%d)\n", kernel_name[k], (int)r);
			return 0;
		}
	}
	return 1;
}

static void wait_ours(device_t *d, sim_t *s)
{
	for (uint32_t i = 0; i < s->nimg; i++)
		if (s->used[i]) {
			d->WaitForFences(d->handle, 1, &s->fence[i], VK_TRUE, 5000000000ull);
			s->used[i] = 0;
		}
}

/* the depth copy through the game's graphics queue (a present family without graphics), after wait_ours:
 * no work of ours waits on a gsem any more */
static void g_release(device_t *d, sim_t *s)
{
	for (uint32_t i = 0; i < MAX_IMAGES; i++) {
		if (s->gused[i] && s->gfence[i]) d->WaitForFences(d->handle, 1, &s->gfence[i], VK_TRUE, 5000000000ull);
		if (s->gfence[i]) d->DestroyFence(d->handle, s->gfence[i], NULL);
		if (s->gsem[i]) d->DestroySemaphore(d->handle, s->gsem[i], NULL);
		free_buf(d, &s->gstage[i]);
		s->gfence[i] = VK_NULL_HANDLE, s->gsem[i] = VK_NULL_HANDLE, s->gcmd[i] = VK_NULL_HANDLE;
		s->gused[i] = s->gsig[i] = 0, s->gbytes[i] = 0;
	}
	if (s->gpool) d->DestroyCommandPool(d->handle, s->gpool, NULL);
	s->gpool = VK_NULL_HANDLE, s->gqueue = VK_NULL_HANDLE, s->gfamily = 0;
}

/* all 1: the device goes, so the pipelines too; 0: only what the swapchain sized; 2: as 0, but the
 * simulator moves to another queue family of the same swapchain, so the semaphores stay (a present
 * queued on the old queue may still wait on one, and they serve on any queue) */
static void sim_release(device_t *d, sim_t *s, int all)
{
	wait_ours(d, s);
	g_release(d, s);
	for (uint32_t i = 0; i < MAX_IMAGES; i++) {
		if (s->fence[i]) d->DestroyFence(d->handle, s->fence[i], NULL);
		if (s->sem[i] && all != 2) d->DestroySemaphore(d->handle, s->sem[i], NULL), s->sem[i] = VK_NULL_HANDLE;
		s->fence[i] = VK_NULL_HANDLE, s->cmd[i] = VK_NULL_HANDLE;
		s->rec_at[i] = s->stat_at[i] = s->snap_at[i] = 0;
	}
	if (s->qpool) d->DestroyQueryPool(d->handle, s->qpool, NULL);
	s->qpool = VK_NULL_HANDLE, s->ts_ok = 0;
	for (uint32_t i = 0; i < MAX_IMAGES; i++) s->ts_at[i] = 0;
	if (s->pool) d->DestroyCommandPool(d->handle, s->pool, NULL);
	s->pool = VK_NULL_HANDLE;
	if (s->dpool) d->DestroyDescriptorPool(d->handle, s->dpool, NULL);
	s->dpool = VK_NULL_HANDLE;
	for (int i = 0; i < MAX_SLOTS; i++) {
		free_buf(d, &s->color[i]);
		free_buf(d, &s->depth[i]);
	}
	free_buf(d, &s->zbuf), free_buf(d, &s->out), free_buf(d, &s->fin), free_buf(d, &s->pkeys);
	free_buf(d, &s->pcolor), free_buf(d, &s->params), free_buf(d, &s->rec), free_buf(d, &s->snap);
	s->ready = 0, s->sc = NULL, s->frames = 0, s->snap_busy = 0;
	s->log_frames = 0;                       /* else the next fps line counts from the old frame number */
	if (s->rec_data) {
		free(s->rec_data), free(s->rec_qpc), free(s->rec_mode);
		s->rec_data = NULL, s->rec_qpc = NULL, s->rec_mode = NULL;
		s->rec_left = s->rec_total = s->rec_got = 0;
		say("sim: the recording was cut short by a new swapchain\n");
	}
	for (uint32_t i = 0; i < MAX_IMAGES; i++) s->demo_at[i] = 0;
	if (s->demo_total) {
		say("sim: the demo recording is cut short by a new swapchain; what was captured is kept\n");
		demo_finish(s);
	}
	if (all == 1) {
		for (int k = 0; k < K_COUNT; k++)
			if (s->pipe[k]) d->DestroyPipeline(d->handle, s->pipe[k], NULL), s->pipe[k] = VK_NULL_HANDLE;
		if (s->pl) d->DestroyPipelineLayout(d->handle, s->pl, NULL), s->pl = VK_NULL_HANDLE;
		if (s->dsl) d->DestroyDescriptorSetLayout(d->handle, s->dsl, NULL), s->dsl = VK_NULL_HANDLE;
		s->pipes = 0;
		if (s->shm) UnmapViewOfFile(s->shm), s->shm = NULL;
		if (s->mapping) CloseHandle(s->mapping), s->mapping = NULL;
		if (s->dump_event) CloseHandle(s->dump_event), s->dump_event = NULL;
		if (s->keys_shm) UnmapViewOfFile((LPCVOID)s->keys_shm), s->keys_shm = NULL;
		if (s->keys_map && s->keys_map != INVALID_HANDLE_VALUE) CloseHandle(s->keys_map);
		s->keys_map = NULL;
	}
}

/* ---------------------------------------------------------------- the mouse */

struct launch { wchar_t exe[MAX_PATH + 32], line[3 * MAX_PATH + 160]; };

static DWORD WINAPI launch_thread(LPVOID p)
{
	struct launch *l = p;
	STARTUPINFOW si = { sizeof si };
	PROCESS_INFORMATION pi;
	if (CreateProcessW(l->exe, l->line, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
		say("sim: input_logger started (pid %lu)\n", pi.dwProcessId);
		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
	} else {
		say("sim: input_logger did not start (error %lu)\n", GetLastError());
	}
	free(l);
	return 0;
}

static void mouse_start(sim_t *s)
{
	wchar_t name[96];
	_snwprintf(name, 96, L"Local\\cloud-sim-vk-mouse-%lu", GetCurrentProcessId());
	name[95] = 0;
	s->mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)sizeof(struct mouse_shm), name);
	s->shm = s->mapping ? MapViewOfFile(s->mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(struct mouse_shm)) : NULL;
	if (!s->shm) {
		say("sim: no shared memory for the mouse (error %lu); warp turns nothing\n", GetLastError());
		return;
	}
	if (!cfg.logger) {
		say("sim: the mouse from %ls (input_logger not started: sim_logger=0)\n", name);
		return;
	}
	struct launch *l = calloc(1, sizeof *l);
	if (!l) return;
	_snwprintf(l->exe, MAX_PATH + 32, L"%ls\\input_logger.exe", here);
	l->exe[MAX_PATH + 31] = 0;
	_snwprintf(l->line, 3 * MAX_PATH + 160, L"\"%ls\" \"shm:%ls\" - %lu 86400", l->exe, name, GetCurrentProcessId());
	l->line[3 * MAX_PATH + 159] = 0;
	HANDLE t = GetFileAttributesW(l->exe) != INVALID_FILE_ATTRIBUTES ? CreateThread(NULL, 0, launch_thread, l, 0, NULL) : NULL;
	if (t) {
		CloseHandle(t);
		say("sim: the mouse comes from input_logger through %ls\n", name);
	} else {
		say("sim: no input_logger.exe beside the layer; warp turns nothing\n");
		free(l);
	}
}

/* sum over the mouse events of v_i (1 - exp(-(t - t_i) / lag)), t_i <= t, read at t2 minus at t1 */
static void mouse_counts(sim_t *s, LONGLONG t1, LONGLONG t2, double *dx, double *dy)
{
	*dx = *dy = 0.0;
	if (!s->shm) return;
	double lag = cfg.lag_ms * 1e-3 * (double)qpf.QuadPart;
	LONGLONG oldest = t1 - (LONGLONG)(8.0 * lag) - 1;
	int64_t c = s->shm->count;
	for (int64_t i = c - 1; i >= 0 && i > c - MOUSE_SHM_EVENTS + 1; i--) {
		struct mouse_shm_event e = s->shm->ring[i & (MOUSE_SHM_EVENTS - 1)];
		if (e.qpc < oldest) break;
		if (e.qpc > t2) continue;
		double r2 = lag > 0 ? 1.0 - exp(-(double)(t2 - e.qpc) / lag) : 1.0;
		double r1 = e.qpc <= t1 ? (lag > 0 ? 1.0 - exp(-(double)(t1 - e.qpc) / lag) : 1.0) : 0.0;
		*dx += e.dx * (r2 - r1);
		*dy += e.dy * (r2 - r1);
	}
}

/* ---------------------------------------------------------------- the gamepad */

static double pad_table(const float *tab, double m)   /* m in 0..1, linear between the knots */
{
	double x = m * (PAD_KNOTS - 1);
	int i = (int)x;
	if (i >= PAD_KNOTS - 1) return tab[PAD_KNOTS - 1];
	double f = x - i;
	return tab[i] * (1.0 - f) + tab[i + 1] * f;
}

/* the camera's turn rates the right stick asks for (rad/s): about y (yaw) and about x (pitch) */
static void pad_rates(const struct mouse_shm_pad *e, double *ry, double *rp)
{
	*ry = *rp = 0.0;
	if (e->user == MOUSE_SHM_NO_PAD) return;
	double sx = e->rx / 32767.0, sy = e->ry / 32767.0;
	sx = sx < -1.0 ? -1.0 : sx, sy = sy < -1.0 ? -1.0 : sy;
	if (cfg.pad_radial) {
		double m = sqrt(sx * sx + sy * sy);
		if (m < 1e-9) return;
		double mm = m > 1.0 ? 1.0 : m;
		*ry = pad_table(cfg.pad_yaw, mm) * sx / m;
		*rp = pad_table(cfg.pad_pitch, mm) * sy / m;
	} else {
		*ry = pad_table(cfg.pad_yaw, fabs(sx)) * (sx < 0 ? -1.0 : 1.0);
		*rp = pad_table(cfg.pad_pitch, fabs(sy)) * (sy < 0 ? -1.0 : 1.0);
	}
}

/* At time t (QPC), the part of a rate held over [a, b) that has turned the camera: the rate reaches
 * the camera through a first-order lag L (s), and the turn is its integral; seconds. */
static double pad_part(LONGLONG t, LONGLONG a, LONGLONG b, double L)
{
	if (t <= a) return 0.0;
	LONGLONG c = b < t ? b : t;
	double F = (double)qpf.QuadPart, span = (double)(c - a) / F;
	if (L <= 0.0) return span;
	return span - L * (exp(-(double)(t - c) / (L * F)) - exp(-(double)(t - a) / (L * F)));
}

/* The turn the stick gave the camera from t1 to t2 (QPC, both already moved back by the pad's tau):
 * each sample's rates hold until the next sample. wx about x (pitch), wy about y (yaw), radians. */
static void pad_turn(sim_t *s, LONGLONG t1, LONGLONG t2, double *wx, double *wy)
{
	*wx = *wy = 0.0;
	if (!cfg.pad || !s->shm || s->shm->version < 2) return;
	double L = cfg.pad_lag_ms > 0.0f ? cfg.pad_lag_ms * 1e-3 : 0.0;
	LONGLONG oldest = t1 - (LONGLONG)(8.0 * L * (double)qpf.QuadPart) - 1, next = INT64_MAX;
	int64_t c = s->shm->pad_count;
	for (int64_t i = c - 1; i >= 0 && i > c - MOUSE_SHM_PADS + 1; i--) {
		struct mouse_shm_pad e = s->shm->pads[i & (MOUSE_SHM_PADS - 1)];
		LONGLONG a = e.qpc, b = next;
		next = e.qpc;
		if (a >= t2) continue;
		double ry, rp;
		pad_rates(&e, &ry, &rp);
		double g = pad_part(t2, a, b, L) - pad_part(t1, a, b, L);
		*wy += ry * g, *wx += rp * g;
		if (a < oldest) break;                /* the samples before it turned both ends alike */
	}
}

/* The camera's turn from a frame presented at q to one presented at `to`, as the warp predicts it:
 * the mouse (sim_cx, sim_cy through tau and lag) plus the stick (sim_pad); a rotation vector. */
static void predict_turn(sim_t *s, LONGLONG q, LONGLONG to, double w[3])
{
	double mx, my, px, py;
	LONGLONG tau = (LONGLONG)(cfg.tau_ms * 1e-3 * (double)qpf.QuadPart);
	mouse_counts(s, q - tau, to - tau, &mx, &my);
	for (int a = 0; a < 3; a++) w[a] = cfg.cx[a] * mx + cfg.cy[a] * my;
	LONGLONG ptau = (LONGLONG)(cfg.pad_tau_ms * 1e-3 * (double)qpf.QuadPart);
	pad_turn(s, q - ptau, to - ptau, &px, &py);
	w[0] += px, w[1] += py;
}

static void rodrigues(const double w[3], double R[3][3])
{
	double th = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
	double k[3] = { 0, 0, 0 };
	if (th > 1e-12)
		for (int i = 0; i < 3; i++) k[i] = w[i] / th;
	double K[3][3] = { { 0, -k[2], k[1] }, { k[2], 0, -k[0] }, { -k[1], k[0], 0 } };
	double sn = sin(th), c1 = 1.0 - cos(th);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			double kk = 0;
			for (int m = 0; m < 3; m++) kk += K[i][m] * K[m][j];
			R[i][j] = (i == j) + sn * K[i][j] + c1 * kk;
		}
}

/* ---------------------------------------------------------------- files */

typedef struct job { int n; wchar_t path[6][MAX_PATH + 64]; void *data[6]; size_t size[6]; } job_t;

static DWORD WINAPI write_thread(LPVOID p)
{
	job_t *j = p;
	for (int i = 0; i < j->n; i++) {
		FILE *f = _wfopen(j->path[i], L"wb");
		size_t put = f ? fwrite(j->data[i], 1, j->size[i], f) : 0;
		if (f) fclose(f);
		say("wrote %ls (%zu of %zu bytes)\n", j->path[i], put, j->size[i]);
		free(j->data[i]);
	}
	free(j);
	return 0;
}

static void job_add(job_t *j, const wchar_t *dir, const wchar_t *name, void *data, size_t size)
{
	if (j->n >= 6 || !data) {
		free(data);
		return;
	}
	_snwprintf(j->path[j->n], MAX_PATH + 64, L"%ls\\%ls", dir, name);
	j->path[j->n][MAX_PATH + 63] = 0;
	j->data[j->n] = data, j->size[j->n] = size;
	j->n++;
}

static void job_start(job_t *j)
{
	HANDLE t = CreateThread(NULL, 0, write_thread, j, 0, NULL);
	if (t) CloseHandle(t);
	else write_thread(j);
}

static char *text_dup(const char *s)
{
	size_t n = strlen(s);
	char *t = malloc(n + 1);
	if (t) memcpy(t, s, n + 1);
	return t;
}

/* <rec_dir>\<prefix>-<date>-<time>; 0 when it cannot be made */
static int new_folder(const wchar_t *prefix, wchar_t out[MAX_PATH])
{
	SYSTEMTIME t;
	GetLocalTime(&t);
	CreateDirectoryW(cfg.rec_dir, NULL);
	_snwprintf(out, MAX_PATH, L"%ls\\%ls-%04u%02u%02u-%02u%02u%02u", cfg.rec_dir, prefix, t.wYear, t.wMonth, t.wDay, t.wHour,
		   t.wMinute, t.wSecond);
	out[MAX_PATH - 1] = 0;
	return CreateDirectoryW(out, NULL) || GetLastError() == ERROR_ALREADY_EXISTS;
}

static char *mouse_text(sim_t *s, LONGLONG t0, LONGLONG t1, size_t *size);

static void rec_finish(device_t *d, sim_t *s)
{
	(void)d;
	job_t *j = calloc(1, sizeof *j);
	size_t frame = (size_t)s->wpr * 4 * s->lumh;
	if (!j) return;
	job_add(j, s->rec_path, L"frames.y8", s->rec_data, frame * s->rec_got);
	size_t cap = (size_t)s->rec_got * 64 + 256;
	char *times = malloc(cap);
	if (times) {
		size_t at = (size_t)snprintf(times, cap, "# frame qpc mode keys (W 1, A 2, S 4, D 8, Shift 16, Space 32, Ctrl 64)\n");
		for (int i = 0; i < s->rec_got && at < cap; i++)
			at += (size_t)snprintf(times + at, cap - at, "%d %lld %d %d\n", i, s->rec_qpc[i], s->rec_mode[i] & 0xFF,
					       s->rec_mode[i] >> 8);
		job_add(j, s->rec_path, L"times.txt", times, at < cap ? at : cap);
	}
	LONGLONG t0 = s->rec_qpc[0] - 2 * qpf.QuadPart, t1 = s->rec_qpc[s->rec_got - 1] + qpf.QuadPart / 2;
	size_t isz = 0;
	char *in = mouse_text(s, t0, t1, &isz);
	if (in) job_add(j, s->rec_path, L"input.txt", in, isz);
	char info[1024];
	int m = snprintf(info, sizeof info,
			 "luma_width %u\nluma_height %u\nluma_row_bytes %u\nframes %d\nscreen %u %u\nqpc_frequency %lld\n"
			 "hfov %.4f\ntau_ms %.3f\nlag_ms %.3f\ncx %g %g %g\ncy %g %g %g\nkeep %.4f\n",
			 s->lumw, s->lumh, s->wpr * 4, s->rec_got, s->W, s->H, qpf.QuadPart, cfg.hfov, cfg.tau_ms, cfg.lag_ms,
			 cfg.cx[0], cfg.cx[1], cfg.cx[2], cfg.cy[0], cfg.cy[1], cfg.cy[2], cfg.keep);
	job_add(j, s->rec_path, L"info.txt", text_dup(info), (size_t)(m > 0 ? m : 0));
	say("sim: recording of %d frames done; writing %ls\n", s->rec_got, s->rec_path);
	job_start(j);
	free(s->rec_qpc), free(s->rec_mode);
	s->rec_data = NULL, s->rec_qpc = NULL, s->rec_mode = NULL;
	s->rec_total = s->rec_got = 0;
}

static void snap_finish(sim_t *s)
{
	wchar_t dir[MAX_PATH];
	DWORD n = s->snap_kind == 2 ? GetEnvironmentVariableW(L"CLOUD_SIM_VK_DUMP", dir, MAX_PATH) : 0;
	if (n && n < MAX_PATH) {
		CreateDirectoryW(dir, NULL);         /* the test's fixed folder */
	} else if (!new_folder(s->snap_kind == 1 ? L"snap" : L"dump", dir)) {
		say("sim: no folder for the %s\n", s->snap_kind == 1 ? "snapshot" : "dump");
		return;
	}
	const wchar_t *at = dir;
	job_t *j = calloc(1, sizeof *j);
	if (!j) return;
	unsigned char *a = malloc(s->snap_bytes0), *b = malloc(s->snap_bytes1 ? s->snap_bytes1 : 1);
	if (a) memcpy(a, s->snap.map, s->snap_bytes0);
	if (b && s->snap_bytes1) memcpy(b, (unsigned char *)s->snap.map + (size_t)s->W * s->H * 4, s->snap_bytes1);
	if (s->snap_kind == 1) {
		job_add(j, at, L"color.raw", a, s->snap_bytes0);
		if (s->snap_bytes1) job_add(j, at, L"depth.raw", b, s->snap_bytes1);
		else free(b);
		job_add(j, at, L"snapshot.txt", text_dup(s->snap_info), strlen(s->snap_info));
	} else {
		job_add(j, at, L"out.raw", a, s->snap_bytes0);
		job_add(j, at, L"src.raw", b, s->snap_bytes1);
		job_add(j, at, L"dump.txt", text_dup(s->snap_info), strlen(s->snap_info));
	}
	say("sim: %s to %ls\n", s->snap_kind == 1 ? "snapshot" : "dump", at);
	job_start(j);
}

/* the depth grid: its spread, and the far value's side when sim_depth_reversed=auto */
static void stat_finish(sim_t *s, const uint32_t *v, uint32_t flags)
{
	static float d[STAT_WORDS];
	int n = 0, zeros = 0, ones = 0;
	for (int i = 0; i < STAT_WORDS; i++) {
		if (v[i] == 0xFFFFFFFFu) continue;
		float x;
		if (flags & 4) x = (float)(v[i] & 0xFFFF) / 65535.0f;
		else if (flags & 1) x = (float)(v[i] & 0xFFFFFF) / 16777215.0f;
		else memcpy(&x, &v[i], 4);
		if (!(x >= 0.0f)) x = 0.0f;
		if (x == 0.0f) zeros++;
		if (x >= 1.0f) ones++;
		d[n++] = x;
	}
	if (!n) return;
	/* a few order statistics, by sorting (2560 values a second) */
	for (int i = 1; i < n; i++) {
		float x = d[i];
		int j = i - 1;
		while (j >= 0 && d[j] > x) d[j + 1] = d[j], j--;
		d[j + 1] = x;
	}
	int vote = ones > zeros + 40 ? 0 : zeros > ones + 40 ? 1 : d[n / 2] > 0.5f ? 0 : 1;
	if (cfg.reversed < 0) {
		if (s->reversed < 0) {
			s->reversed = vote;
			say("sim: depth looks %s (zeros %d, ones %d, median %.6f)\n", vote ? "reversed (1 near)" : "standard (0 near)",
			    zeros, ones, d[n / 2]);
		} else if (vote != s->reversed) {
			if (++s->rev_against >= 5) {
				s->reversed = vote, s->rev_against = 0;
				say("sim: depth now looks %s\n", vote ? "reversed" : "standard");
			}
		} else {
			s->rev_against = 0;
		}
	}
	say("sim: depth grid: %d samples, 0.0 x%d, 1.0 x%d, min %.6f p1 %.6f p5 %.6f p25 %.6f p50 %.6f p75 %.6f p95 %.6f max %.6f\n", n,
	    zeros, ones, d[0], d[n / 100], d[n / 20], d[n / 4], d[n / 2], d[3 * n / 4], d[19 * n / 20], d[n - 1]);
}

/* ---------------------------------------------------------------- the demo recording */

/* Insert (or sim_demo_now): sim_demo_frames frames of half-size colour (raw RGBA8/BGRA8) and
 * nearness (16-bit, 0 at the camera, 65535 far), streamed to color.half and near.half by a writer
 * thread (about 3 MB a frame at 1080p: ~250 MB/s at 85 fps; a frame the queue has no room for is
 * dropped and counted), then times.txt (the frames written), input.txt (the mouse), info.txt (the
 * sizes and the simulator's settings). doom/demo_doom.py makes the video from it. */
#define DEMO_QUEUE 64
typedef struct { unsigned char *data; size_t cbytes, dbytes; } dframe_t;
static struct {
	SRWLOCK lock;
	CONDITION_VARIABLE cv;
	dframe_t q[DEMO_QUEUE];
	int head, count, finish, busy, dropped, written;
	FILE *fc, *fd;
	job_t *texts;
	wchar_t dir[MAX_PATH];
} dw = { SRWLOCK_INIT, CONDITION_VARIABLE_INIT };
static LONGLONG *demo_qpc;
static int *demo_mode, demo_accepted;

static DWORD WINAPI demo_writer(LPVOID unused)
{
	(void)unused;
	for (;;) {
		AcquireSRWLockExclusive(&dw.lock);
		while (!dw.count && !dw.finish) SleepConditionVariableSRW(&dw.cv, &dw.lock, INFINITE, 0);
		if (!dw.count) {                     /* finish, and nothing left to write */
			job_t *texts = dw.texts;
			FILE *fc = dw.fc, *fd = dw.fd;
			int written = dw.written, dropped = dw.dropped;
			dw.texts = NULL, dw.fc = dw.fd = NULL, dw.finish = 0;
			ReleaseSRWLockExclusive(&dw.lock);
			if (fc) fclose(fc);
			if (fd) fclose(fd);
			say("sim: demo recording: %d frames written, %d dropped (the disk fell behind), into %ls\n", written, dropped, dw.dir);
			if (texts) write_thread(texts);
			AcquireSRWLockExclusive(&dw.lock);
			dw.busy = 0;
			ReleaseSRWLockExclusive(&dw.lock);
			return 0;
		}
		dframe_t f = dw.q[dw.head];
		dw.head = (dw.head + 1) % DEMO_QUEUE;
		dw.count--;
		ReleaseSRWLockExclusive(&dw.lock);
		if (dw.fc) fwrite(f.data, 1, f.cbytes, dw.fc);
		if (dw.fd) fwrite(f.data + f.cbytes, 1, f.dbytes, dw.fd);
		free(f.data);
		AcquireSRWLockExclusive(&dw.lock);
		dw.written++;
		ReleaseSRWLockExclusive(&dw.lock);
	}
}

/* The mouse events from t0 to t1 in input_logger's format, then the gamepad's samples from the last
 * one before t0 to t1, one a line:
 *   p <QPC> <left x> <left y> <right x> <right y> <left trigger> <right trigger> <buttons, hex> <user>
 * (the sticks raw, -32768..32767, y up; user -1: the pad went away). malloc'd, *size its length. */
static char *mouse_text(sim_t *s, LONGLONG t0, LONGLONG t1, size_t *size)
{
	size_t icap = 8u << 20;
	char *in = malloc(icap);
	*size = 0;
	if (!in || !s->shm) {
		free(in);
		return NULL;
	}
	size_t at = (size_t)snprintf(in, icap, "# input_logger 1\n# frequency %lld\n# start %lld\n", qpf.QuadPart, t0);
	int64_t c = s->shm->count, first = c - MOUSE_SHM_EVENTS + 2, n = 0, np = 0;
	if (first < 0) first = 0;
	for (int64_t i = first; i < c && at + 96 < icap; i++) {
		struct mouse_shm_event e = s->shm->ring[i & (MOUSE_SHM_EVENTS - 1)];
		if (e.qpc < t0 || e.qpc > t1) continue;
		at += (size_t)snprintf(in + at, icap - at, "m %lld %d %d 0 0 1 0\n", e.qpc, e.dx, e.dy);
		n++;
	}
	if (s->shm->version >= 2) {
		int64_t pc = s->shm->pad_count, pf = pc - MOUSE_SHM_PADS + 2;
		if (pf < 0) pf = 0;
		for (int64_t i = pf; i < pc && at + 128 < icap; i++) {
			struct mouse_shm_pad e = s->shm->pads[i & (MOUSE_SHM_PADS - 1)];
			if (e.qpc > t1) break;
			if (e.qpc < t0 && i + 1 < pc && s->shm->pads[(i + 1) & (MOUSE_SHM_PADS - 1)].qpc < t0) continue;
			at += (size_t)snprintf(in + at, icap - at, "p %lld %d %d %d %d %u %u %x %d\n", e.qpc, e.lx, e.ly, e.rx, e.ry,
					       (unsigned)e.lt, (unsigned)e.rt, (unsigned)e.buttons,
					       e.user == MOUSE_SHM_NO_PAD ? -1 : (int)e.user);
			np++;
		}
	}
	at += (size_t)snprintf(in + at, icap - at, "# end %lld %lld 0 layer (%lld pad samples)\n", t1, n, np);
	*size = at;
	return in;
}

static void demo_start(sim_t *s)
{
	if (!s->demo_ok) {
		say("sim: no demo recording at %ux%u (the width must be a multiple of 4, the height even)\n", s->W, s->H);
		return;
	}
	AcquireSRWLockExclusive(&dw.lock);
	int busy = dw.busy;
	ReleaseSRWLockExclusive(&dw.lock);
	if (busy || s->demo_total) {
		say("sim: a demo recording is still running or being written\n");
		return;
	}
	if (!new_folder(L"demo", dw.dir)) {
		say("sim: no folder for the demo recording under %ls\n", cfg.rec_dir);
		return;
	}
	wchar_t pc[MAX_PATH + 32], pd[MAX_PATH + 32];
	_snwprintf(pc, MAX_PATH + 32, L"%ls\\color.half", dw.dir);
	_snwprintf(pd, MAX_PATH + 32, L"%ls\\near.half", dw.dir);
	pc[MAX_PATH + 31] = pd[MAX_PATH + 31] = 0;
	dw.fc = _wfopen(pc, L"wb");
	dw.fd = _wfopen(pd, L"wb");
	demo_qpc = calloc((size_t)cfg.demo_frames, sizeof *demo_qpc);
	demo_mode = calloc((size_t)cfg.demo_frames, sizeof *demo_mode);
	if (!dw.fc || !dw.fd || !demo_qpc || !demo_mode) {
		if (dw.fc) fclose(dw.fc);
		if (dw.fd) fclose(dw.fd);
		dw.fc = dw.fd = NULL;
		free(demo_qpc), free(demo_mode);
		demo_qpc = NULL, demo_mode = NULL;
		say("sim: the demo recording's files could not be opened in %ls\n", dw.dir);
		return;
	}
	setvbuf(dw.fc, NULL, _IOFBF, 1 << 22);
	setvbuf(dw.fd, NULL, _IOFBF, 1 << 21);
	dw.head = dw.count = dw.finish = dw.dropped = dw.written = 0;
	dw.busy = 1;
	HANDLE t = CreateThread(NULL, 0, demo_writer, NULL, 0, NULL);
	if (!t) {
		fclose(dw.fc), fclose(dw.fd);
		dw.fc = dw.fd = NULL, dw.busy = 0;
		say("sim: no writer thread for the demo recording\n");
		return;
	}
	CloseHandle(t);
	demo_accepted = 0;
	s->demo_total = s->demo_left = cfg.demo_frames;
	s->demo_got = 0;
	say("sim: demo recording of %d frames (%ux%u colour + nearness, %.1f MB a frame) into %ls\n", s->demo_total, s->W / 2, s->H / 2,
	    (s->demo_cwords + s->demo_dwords) * 4 / 1e6, dw.dir);
}

static void demo_push(unsigned char *data, size_t cbytes, size_t dbytes, LONGLONG q, int mode, int total)
{
	AcquireSRWLockExclusive(&dw.lock);
	int ok = dw.count < DEMO_QUEUE && demo_accepted < total;
	if (ok) {
		dw.q[(dw.head + dw.count) % DEMO_QUEUE] = (dframe_t){ data, cbytes, dbytes };
		dw.count++;
	} else {
		dw.dropped++;
	}
	ReleaseSRWLockExclusive(&dw.lock);
	if (ok) {
		WakeConditionVariable(&dw.cv);
		demo_qpc[demo_accepted] = q, demo_mode[demo_accepted] = mode;
		demo_accepted++;
	} else {
		free(data);
	}
}

static void demo_finish(sim_t *s)
{
	job_t *j = calloc(1, sizeof *j);
	if (j && demo_accepted) {
		size_t cap = (size_t)demo_accepted * 64 + 256;
		char *times = malloc(cap);
		if (times) {
			size_t at = (size_t)snprintf(times, cap,
						     "# frame qpc mode keys (the frames written, in order; W 1, A 2, S 4, D 8, Shift 16, "
						     "Space 32, Ctrl 64)\n");
			for (int i = 0; i < demo_accepted && at < cap; i++)
				at += (size_t)snprintf(times + at, cap - at, "%d %lld %d %d\n", i, demo_qpc[i], demo_mode[i] & 0xFF,
						       demo_mode[i] >> 8);
			job_add(j, dw.dir, L"times.txt", times, at < cap ? at : cap);
		}
		size_t isz = 0;
		char *in = mouse_text(s, demo_qpc[0] - 2 * qpf.QuadPart, demo_qpc[demo_accepted - 1] + qpf.QuadPart / 2, &isz);
		if (in) job_add(j, dw.dir, L"input.txt", in, isz);
		char info[2048];
		int n = snprintf(info, sizeof info,
				 "half_width %u\nhalf_height %u\nframes %d\nscreen %u %u\nbgra %u\nqpc_frequency %lld\n"
				 "nearness 16-bit, 0 at the camera, 65535 far (the nearest of each 2x2)\n"
				 "hfov %.4f\ntau_ms %.3f\nlag_ms %.3f\ncx %g %g %g\ncy %g %g %g\nkeep %.4f\nweapon_depth %g\nreversed %d\n"
				 "hole %.4f\ncross %.4f\n"
				 "lock1 %g %g %g %g\nlock2 %g %g %g %g\nlock3 %g %g %g %g\nlock4 %g %g %g %g\n",
				 s->W / 2, s->H / 2, demo_accepted, s->W, s->H, s->bgra, qpf.QuadPart, cfg.hfov, cfg.tau_ms, cfg.lag_ms,
				 cfg.cx[0], cfg.cx[1], cfg.cx[2], cfg.cy[0], cfg.cy[1], cfg.cy[2], cfg.keep, cfg.weapon,
				 cfg.reversed >= 0 ? cfg.reversed : (s->reversed > 0), cfg.hole, cfg.cross, cfg.lock[0][0], cfg.lock[0][1],
				 cfg.lock[0][2], cfg.lock[0][3], cfg.lock[1][0], cfg.lock[1][1], cfg.lock[1][2], cfg.lock[1][3], cfg.lock[2][0],
				 cfg.lock[2][1], cfg.lock[2][2], cfg.lock[2][3], cfg.lock[3][0], cfg.lock[3][1], cfg.lock[3][2], cfg.lock[3][3]);
		job_add(j, dw.dir, L"info.txt", text_dup(info), (size_t)(n > 0 ? n : 0));
	}
	AcquireSRWLockExclusive(&dw.lock);
	dw.texts = j;
	dw.finish = 1;
	ReleaseSRWLockExclusive(&dw.lock);
	WakeConditionVariable(&dw.cv);
	say("sim: demo recording done: %d frames captured, %d queued for the disk\n", s->demo_got, demo_accepted);
	free(demo_qpc), free(demo_mode);
	demo_qpc = NULL, demo_mode = NULL;
	s->demo_total = s->demo_got = s->demo_left = 0;
}

static void collect(device_t *d, sim_t *s, uint32_t i)
{
	(void)d;
	uint32_t *base = (uint32_t *)s->rec.map + (size_t)i * s->rec_stride;
	if (s->demo_at[i]) {
		s->demo_at[i] = 0;
		if (s->demo_total && s->demo_got < s->demo_total) {
			size_t cb = (size_t)s->demo_cwords * 4, db = (size_t)s->demo_dwords * 4;
			unsigned char *data = malloc(cb + db);
			if (data) {
				memcpy(data, base + s->demo_off, cb);
				memcpy(data + cb, base + s->demo_doff, db);
				demo_push(data, cb, db, s->demo_q[i], s->demo_m[i], s->demo_total);
			}
			if (++s->demo_got == s->demo_total) demo_finish(s);
		}
	}
	if (s->rec_at[i]) {
		s->rec_at[i] = 0;
		size_t frame = (size_t)s->wpr * 4 * s->lumh;
		if (s->rec_data && s->rec_got < s->rec_total) {
			memcpy(s->rec_data + frame * s->rec_got, base, frame);
			s->rec_qpc[s->rec_got] = s->rec_q[i];
			s->rec_mode[s->rec_got] = s->rec_m[i];
			if (++s->rec_got == s->rec_total) rec_finish(d, s);
		}
	}
	if (s->stat_at[i]) {
		s->stat_at[i] = 0;
		stat_finish(s, base + s->luma_words, s->stat_fl[i]);
	}
	if (s->snap_at[i]) {
		s->snap_at[i] = 0;
		snap_finish(s);
		s->snap_busy = 0;
	}
}

/* ---------------------------------------------------------------- set up for a swapchain */

static void write_set(device_t *d, VkDescriptorSet set, const VkDescriptorBufferInfo bi[8])
{
	VkWriteDescriptorSet w[8];
	for (uint32_t i = 0; i < 8; i++)
		w[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, i, 0, 1,
					       i == 6 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[i], NULL };
	d->UpdateDescriptorSets(d->handle, 8, w, 0, NULL);
}

static VkDescriptorBufferInfo whole(const buf_t *b) { return (VkDescriptorBufferInfo){ b->b, 0, VK_WHOLE_SIZE }; }

static VkDescriptorBufferInfo keys_at(sim_t *s, uint32_t l)
{
	if (!l) return whole(&s->zbuf);
	return (VkDescriptorBufferInfo){ s->pkeys.b, s->koff[l], (VkDeviceSize)s->lw[l] * s->lh[l] * 4 };
}

static VkDescriptorBufferInfo colors_at(sim_t *s, uint32_t l)
{
	if (!l) return whole(&s->out);
	return (VkDescriptorBufferInfo){ s->pcolor.b, s->coff[l], (VkDeviceSize)s->lw[l] * s->lh[l] * 4 };
}

static int fmt_ok(VkFormat f, uint32_t *bgra)
{
	switch (f) {
	case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
		*bgra = 1;
		return 1;
	case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
	case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
		*bgra = 0;
		return 1;
	default:
		return 0;
	}
}

/* a queue the game got on this device in a family with graphics (the lowest such family), or none */
static VkQueue graphics_queue(device_t *d, const VkQueueFamilyProperties *qf, uint32_t nqf, uint32_t *gf)
{
	VkQueue best = VK_NULL_HANDLE;
	uint32_t bf = UINT32_MAX;
	AcquireSRWLockShared(&obj_lock);
	for (size_t i = 0; i < queues.cap; i++) {
		if (queues.st[i] != 1) continue;
		VkQueue q = (VkQueue)(uintptr_t)queues.k[i];
		uint32_t f = (uint32_t)(uintptr_t)queues.v[i] - 1;
		if (dkey_of(q) != d->key || f >= nqf || !(qf[f].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
		if (f < bf) bf = f, best = q;
	}
	ReleaseSRWLockShared(&obj_lock);
	*gf = bf;
	return best;
}

/* the depth copy through the game's graphics queue gq (family gf): per swapchain image a command buffer,
 * a fence, a semaphore, and a staging buffer both families share; s->family and s->nimg are set */
static int g_init(device_t *d, sim_t *s, VkQueue gq, uint32_t gf, VkDeviceSize cap)
{
	VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, gf };
	if (d->CreateCommandPool(d->handle, &pci, NULL, &s->gpool) != VK_SUCCESS) {
		s->gpool = VK_NULL_HANDLE;
		return 0;
	}
	VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->gpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, s->nimg };
	if (d->AllocateCommandBuffers(d->handle, &cai, s->gcmd) != VK_SUCCESS) return 0;
	for (uint32_t i = 0; i < s->nimg; i++) {
		if (d->set_loader_data) d->set_loader_data(d->handle, s->gcmd[i]);
		else *(void **)s->gcmd[i] = d->key;
		VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		VkSemaphoreCreateInfo si = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		if (d->CreateFence(d->handle, &fi, NULL, &s->gfence[i]) != VK_SUCCESS) {
			s->gfence[i] = VK_NULL_HANDLE;
			return 0;
		}
		if (d->CreateSemaphore(d->handle, &si, NULL, &s->gsem[i]) != VK_SUCCESS) {
			s->gsem[i] = VK_NULL_HANDLE;
			return 0;
		}
		if (!make_buf_in(d, &s->gstage[i], cap, 0, s->family, gf)) return 0;
	}
	s->gqueue = gq, s->gfamily = gf;
	return 1;
}

/* On a present family without graphics, under q_lock: the game's depth (in layout dl) copied on its
 * graphics queue into gstage[i], after all the game has submitted there (the first barrier's first scope)
 * and before all it submits later (the second barrier's second scope: its next frame's depth waits for
 * the copy), and signalled to our work on the present queue by gsem[i]. 1 when it is on its way. */
static int gcopy(device_t *d, sim_t *s, uint32_t i, img_t *dimg, VkImageLayout dl)
{
	if (s->gused[i]) {
		if (d->WaitForFences(d->handle, 1, &s->gfence[i], VK_TRUE, 2000000000ull) != VK_SUCCESS) return 0;
		s->gused[i] = 0;
	}
	if (s->gsig[i]) {               /* signalled and never waited (our frame stopped after its copy): a new one */
		d->DestroySemaphore(d->handle, s->gsem[i], NULL);
		s->gsem[i] = VK_NULL_HANDLE, s->gsig[i] = 0;
		VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		if (d->CreateSemaphore(d->handle, &sci, NULL, &s->gsem[i]) != VK_SUCCESS) s->gsem[i] = VK_NULL_HANDLE;
	}
	if (!s->gsem[i] || !s->gcmd[i]) return 0;
	uint32_t w = dimg->extent.width, h = dimg->extent.height;
	VkDeviceSize bytes = (VkDeviceSize)w * h * (depth_kind(dimg->format) == 4 ? 2 : 4);
	if (bytes > s->gstage[i].size) return 0;
	VkCommandBuffer cb = s->gcmd[i];
	VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
	if (d->BeginCommandBuffer(cb, &bi) != VK_SUCCESS) return 0;
	VkImageMemoryBarrier to = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL,
				    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
				    VK_ACCESS_TRANSFER_READ_BIT, dl, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
				    VK_QUEUE_FAMILY_IGNORED, dimg->image, { dimg->aspect, 0, 1, 0, 1 } };
	d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to);
	VkBufferImageCopy dc = { 0, 0, 0, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 }, { 0, 0, 0 }, { w, h, 1 } };
	d->CmdCopyImageToBuffer(cb, dimg->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s->gstage[i].b, 1, &dc);
	VkImageMemoryBarrier back = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_READ_BIT,
				      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dl,
				      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, dimg->image, { dimg->aspect, 0, 1, 0, 1 } };
	d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &back);
	if (d->EndCommandBuffer(cb) != VK_SUCCESS) return 0;
	d->ResetFences(d->handle, 1, &s->gfence[i]);
	VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 1, &s->gsem[i] };
	if (d->QueueSubmit(s->gqueue, 1, &si, s->gfence[i]) != VK_SUCCESS) return 0;
	s->gused[i] = 1, s->gsig[i] = 1, s->gbytes[i] = bytes;
	return 1;
}

static int sim_init(device_t *d, sim_t *s, VkQueue queue, sc_t *sc)
{
	AcquireSRWLockShared(&obj_lock);
	uint32_t family = (uint32_t)(uintptr_t)map_get(&queues, H64(queue));
	ReleaseSRWLockShared(&obj_lock);
	if (!family) {
		say("sim: the present queue was not got through vkGetDeviceQueue; off\n");
		return 0;
	}
	family--;
	VkQueueFamilyProperties qf[16];
	uint32_t nqf = 16;
	d->inst->GetPhysicalDeviceQueueFamilyProperties(d->phys, &nqf, qf);
	if (family >= nqf || !(qf[family].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT))) {
		say("sim: the present queue's family %u has no compute; off\n", family);
		return 0;
	}
	if (!sc->transfer || !fmt_ok(sc->format, &s->bgra) || !sc->count) {
		say("sim: swapchain format %d / transfer %d not handled; off for this swapchain\n", (int)sc->format, sc->transfer);
		return 0;
	}
	if (!s->pipes) {
		if (!make_pipelines(d, s)) return 0;
		s->pipes = 1;
	}
	if (!s->shm) mouse_start(s);
	if (!s->dump_event) {
		wchar_t name[96];
		_snwprintf(name, 96, L"Local\\cloud-sim-vk-dump-%lu", GetCurrentProcessId());
		name[95] = 0;
		s->dump_event = CreateEventW(NULL, FALSE, FALSE, name);
	}
	s->queue = queue, s->family = family, s->sc = sc;
	/* Vulkan copies depth only on queues with graphics: on a family without, the depth is copied on the
	 * game's own graphics queue (g_init, gcopy), or, with none of those, the simulator runs without it */
	s->graphics = (qf[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
	s->W = sc->extent.width, s->H = sc->extent.height, s->nimg = sc->count, s->slots = (uint32_t)cfg.slots;
	if (!s->graphics && cfg.depth) {
		uint32_t gf = 0;
		VkQueue gq = graphics_queue(d, qf, nqf, &gf);
		if (gq && g_init(d, s, gq, gf, (VkDeviceSize)s->W * s->H * 4)) {
			say("sim: family %u has no graphics: the game's depth is copied on its graphics queue (family %u) and handed "
			    "over by a semaphore\n", family, gf);
		} else {
			g_release(d, s);
			say("sim: family %u has no graphics (%s): the game's depth is left out there (no weapon mask: the weapon turns "
			    "with the world; no depth for the move)\n", family,
			    gq ? "the copy through its graphics queue could not be set up" : "and no graphics queue of the game's");
		}
	}
	VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family };
	if (d->CreateCommandPool(d->handle, &pci, NULL, &s->pool) != VK_SUCCESS) return 0;
	s->ts_ok = 0, s->ts_sum_ms = s->ts_max_ms = 0.0, s->ts_n = 0, s->ts_stride = 4 + PROF_MAX, s->prof_img = -1;
	if (qf[family].timestampValidBits && d->props.limits.timestampPeriod > 0.0f && d->CreateQueryPool && d->CmdWriteTimestamp) {
		VkQueryPoolCreateInfo qi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, s->ts_stride * s->nimg, 0 };
		s->ts_ok = d->CreateQueryPool(d->handle, &qi, NULL, &s->qpool) == VK_SUCCESS;
		s->ts_bits = qf[family].timestampValidBits;
	}
	VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, s->nimg };
	if (d->AllocateCommandBuffers(d->handle, &cai, s->cmd) != VK_SUCCESS) return 0;
	for (uint32_t i = 0; i < s->nimg; i++) {
		if (d->set_loader_data) d->set_loader_data(d->handle, s->cmd[i]);
		else *(void **)s->cmd[i] = d->key;
		VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		VkSemaphoreCreateInfo si = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		if (d->CreateFence(d->handle, &fi, NULL, &s->fence[i]) != VK_SUCCESS) return 0;
		if (!s->sem[i] && d->CreateSemaphore(d->handle, &si, NULL, &s->sem[i]) != VK_SUCCESS) return 0;   /* kept on a move */
	}
	VkDeviceSize frame = (VkDeviceSize)s->W * s->H * 4;
	VkDeviceSize align = d->props.limits.minStorageBufferOffsetAlignment;
	if (align < 256) align = 256;
	s->lw[0] = s->W, s->lh[0] = s->H, s->levels = 0;
	VkDeviceSize kb = 0, cb = 0;
	while (s->levels < LEVELS && (s->lw[s->levels] > 8 || s->lh[s->levels] > 8)) {
		uint32_t l = ++s->levels;
		s->lw[l] = (s->lw[l - 1] + 1) / 2, s->lh[l] = (s->lh[l - 1] + 1) / 2;
		s->koff[l] = kb, s->coff[l] = cb;
		kb += ((VkDeviceSize)s->lw[l] * s->lh[l] * 4 + align - 1) / align * align;
		cb += ((VkDeviceSize)s->lw[l] * s->lh[l] * 4 + align - 1) / align * align;
	}
	s->depth_cap = frame;
	s->lumw = s->W / 4, s->lumh = s->H / 4, s->wpr = (s->lumw + 3) / 4;
	s->luma_words = s->wpr * s->lumh;
	s->demo_ok = s->W % 4 == 0 && s->H % 2 == 0;
	s->demo_cwords = s->demo_ok ? (s->W / 2) * (s->H / 2) : 0;
	s->demo_dwords = s->demo_ok ? (s->W / 4) * (s->H / 2) : 0;
	s->demo_off = (s->luma_words + STAT_WORDS + 63) / 64 * 64;
	s->demo_doff = s->demo_off + (s->demo_cwords + 63) / 64 * 64;
	s->rec_stride = (s->demo_doff + s->demo_dwords + 63) / 64 * 64;
	int ok = 1;
	for (uint32_t k = 0; k < s->slots && ok; k++) ok = make_buf(d, &s->color[k], frame, 0) && make_buf(d, &s->depth[k], s->depth_cap, 0);
	if (s->gqueue && s->depth_cap > (VkDeviceSize)s->W * s->H * 4) ok = 0;   /* gstage is sized as depth_cap */
	ok = ok && make_buf(d, &s->zbuf, frame, 0) && make_buf(d, &s->out, frame, 0) && make_buf(d, &s->fin, frame, 0);
	ok = ok && make_buf(d, &s->pkeys, kb, 0) && make_buf(d, &s->pcolor, cb, 0) && make_buf(d, &s->params, PARAMS_BYTES, 0);
	ok = ok && make_buf(d, &s->rec, (VkDeviceSize)s->rec_stride * 4 * s->nimg, 1) && make_buf(d, &s->snap, frame * 2, 1);
	if (!ok) {
		say("sim: the ring (%u x %.1f MB) could not be made; off\n", s->slots, 2.0 * frame / 1e6);
		return 0;
	}
	uint32_t nsets = s->slots + s->levels + 1;
	VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nsets * 7 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nsets } };
	VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, nsets, 2, ps };
	if (d->CreateDescriptorPool(d->handle, &dpi, NULL, &s->dpool) != VK_SUCCESS) return 0;
	VkDescriptorSetLayout layouts[MAX_SLOTS + LEVELS + 1];
	VkDescriptorSet sets[MAX_SLOTS + LEVELS + 1];
	for (uint32_t i = 0; i < nsets; i++) layouts[i] = s->dsl;
	VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, s->dpool, nsets, layouts };
	if (d->AllocateDescriptorSets(d->handle, &dai, sets) != VK_SUCCESS) return 0;
	VkDescriptorBufferInfo bi[8];
	for (uint32_t k = 0; k < s->slots; k++) {
		s->set_frame[k] = sets[k];
		bi[0] = whole(&s->color[k]), bi[1] = whole(&s->depth[k]), bi[2] = whole(&s->zbuf), bi[3] = whole(&s->out);
		bi[4] = keys_at(s, s->levels ? 1 : 0), bi[5] = colors_at(s, s->levels ? 1 : 0), bi[6] = whole(&s->params), bi[7] = whole(&s->rec);
		write_set(d, sets[k], bi);
	}
	for (uint32_t l = 0; l < s->levels; l++) {
		s->set_level[l] = sets[s->slots + l];
		bi[0] = whole(&s->color[0]), bi[1] = whole(&s->depth[0]), bi[2] = keys_at(s, l), bi[3] = colors_at(s, l);
		bi[4] = keys_at(s, l + 1), bi[5] = colors_at(s, l + 1), bi[6] = whole(&s->params), bi[7] = whole(&s->rec);
		write_set(d, s->set_level[l], bi);
	}
	s->set_zoom = sets[s->slots + s->levels];
	bi[0] = whole(&s->color[0]), bi[1] = whole(&s->depth[0]), bi[2] = whole(&s->zbuf), bi[3] = whole(&s->out);
	bi[4] = whole(&s->zbuf), bi[5] = whole(&s->fin), bi[6] = whole(&s->params), bi[7] = whole(&s->rec);
	write_set(d, s->set_zoom, bi);
	s->frames = 0, s->dimg = NULL, s->reversed = -1, s->rev_against = 0, s->short_ring_said = 0;
	{
		LARGE_INTEGER seed;
		QueryPerformanceCounter(&seed);
		s->rng = (uint64_t)seed.QuadPart ^ 0x9E3779B97F4A7C15ULL;
	}
	s->jit = 0.0, s->lost_count = 0;
	memset(s->lost, 0, sizeof s->lost);
	s->ready = 1;
	say("sim: on for %ux%u (format %d, %s), %u images, ring of %u frames (%.0f MB), fill pyramid of %u levels; "
	    "mode %s, delay %.0f ms; F6 mode, F7 delay, F8 blind, End record, Home snapshot\n",
	    s->W, s->H, (int)sc->format, s->bgra ? "BGRA" : "RGBA", s->nimg, s->slots, (2.0 * s->slots + 3.0) * frame / 1e6,
	    s->levels, mode_name[cfg.mode], cfg.latency_ms);
	return 1;
}

/* ---------------------------------------------------------------- each present */

/* the movement keys now: a test's stand-in if there is one, else the keyboard while the game is in front */
static int game_in_front(void);

static uint32_t sample_keys(sim_t *s)
{
	if (!s->keys_map) {
		wchar_t name[96];
		_snwprintf(name, 96, L"Local\\cloud-sim-vk-keys-%lu", GetCurrentProcessId());
		name[95] = 0;
		s->keys_map = OpenFileMappingW(FILE_MAP_READ, FALSE, name);
		if (!s->keys_map) s->keys_map = INVALID_HANDLE_VALUE;
		else s->keys_shm = MapViewOfFile(s->keys_map, FILE_MAP_READ, 0, 0, sizeof(LONG));
	}
	if (s->keys_shm) return (uint32_t)*s->keys_shm;
	if (!game_in_front()) return 0;
	static const int vk[7] = { 'W', 'A', 'S', 'D', VK_SHIFT, VK_SPACE, VK_CONTROL };
	uint32_t bits = 0;
	for (int i = 0; i < 7; i++)
		if (GetAsyncKeyState(vk[i]) & 0x8000) bits |= 1u << i;
	return bits;
}

/* The camera's move over (a, b], in the old frame's axes (x right, z forward), in near-plane
 * units: each present's keys held since the present before (W forward, S back, D right, A left,
 * a diagonal as fast as a straight line), at sim_move_speed. The turn during the span is left
 * out (a few degrees over 100 ms). */
static void move_over(sim_t *s, LONGLONG a, LONGLONG b, double T[3])
{
	T[0] = T[1] = T[2] = 0.0;
	if (!(b > a) || !(cfg.move_speed > 0.0f)) return;
	unsigned long long older = s->frames < s->slots - 1 ? s->frames : s->slots - 1;
	if (!older) return;
	/* forward in time from the oldest frame, the velocity following the keys' with a first-order
	 * lag (sim_move_lag_ms; 0: at once), taken as steady at the oldest */
	double lag = cfg.move_lag_ms > 0.0f ? cfg.move_lag_ms * 1e-3 : 0.0, v[2] = { 0, 0 };
	for (unsigned long long back = older; back-- > 0;) {
		uint32_t sl = (uint32_t)((s->frames - back) % s->slots), pv = (uint32_t)((s->frames - back - 1) % s->slots);
		uint32_t k = s->keys[sl];
		double fw = ((k & KEY_W) ? 1.0 : 0.0) - ((k & KEY_S) ? 1.0 : 0.0);
		double rt = ((k & KEY_D) ? 1.0 : 0.0) - ((k & KEY_A) ? 1.0 : 0.0);
		double nrm = sqrt(fw * fw + rt * rt);
		double u[2] = { nrm > 0 ? rt / nrm * cfg.move_speed : 0.0, nrm > 0 ? fw / nrm * cfg.move_speed : 0.0 };
		if (back == older - 1) v[0] = u[0], v[1] = u[1];
		double lo = (double)s->qpc[pv] / (double)qpf.QuadPart, hi = (double)s->qpc[sl] / (double)qpf.QuadPart;
		double A = (double)a / (double)qpf.QuadPart, B = (double)b / (double)qpf.QuadPart;
		double o0 = lo > A ? lo : A, o1 = hi < B ? hi : B;
		for (int c = 0; c < 2; c++) {
			if (o1 > o0) {
				double dist = u[c] * (o1 - o0);
				if (lag > 0) dist += (v[c] - u[c]) * lag * (exp(-(o0 - lo) / lag) - exp(-(o1 - lo) / lag));
				T[c == 0 ? 0 : 2] += dist;
			}
			v[c] = lag > 0 ? u[c] + (v[c] - u[c]) * exp(-(hi - lo) / lag) : u[c];
		}
	}
}

static int game_in_front(void)
{
	HWND w = GetForegroundWindow();
	DWORD pid = 0;
	if (w) GetWindowThreadProcessId(w, &pid);
	return pid == GetCurrentProcessId();
}

static void start_recording(sim_t *s)
{
	if (s->rec_left || s->rec_data) return;
	if (!new_folder(L"rec", s->rec_path)) {
		say("sim: no folder for the recording under %ls\n", cfg.rec_dir);
		return;
	}
	size_t frame = (size_t)s->wpr * 4 * s->lumh;
	s->rec_total = cfg.rec_frames;
	s->rec_data = malloc(frame * (size_t)s->rec_total);
	s->rec_qpc = calloc((size_t)s->rec_total, sizeof *s->rec_qpc);
	s->rec_mode = calloc((size_t)s->rec_total, sizeof *s->rec_mode);
	if (!s->rec_data || !s->rec_qpc || !s->rec_mode) {
		free(s->rec_data), free(s->rec_qpc), free(s->rec_mode);
		s->rec_data = NULL, s->rec_qpc = NULL, s->rec_mode = NULL;
		say("sim: no memory for a recording of %d frames\n", s->rec_total);
		return;
	}
	s->rec_left = s->rec_total, s->rec_got = 0;
	say("sim: recording %d frames (%ux%u luma) into %ls\n", s->rec_total, s->lumw, s->lumh, s->rec_path);
}

static int snapshot_wanted, dump_wanted;

static void poll(sim_t *s)
{
	ULONGLONG now = GetTickCount64();
	if (now >= s->next_poll) {
		s->next_poll = now + 500;
		load_settings();
		if (cfg.want_record) cfg.want_record = 0, start_recording(s);
		if (cfg.want_snapshot) cfg.want_snapshot = 0, snapshot_wanted = 1;
		if (cfg.want_dump) cfg.want_dump = 0, dump_wanted = 1;
		if (cfg.want_demo) cfg.want_demo = 0, demo_start(s);
	}
	if (s->dump_event && WaitForSingleObject(s->dump_event, 0) == WAIT_OBJECT_0) dump_wanted = 1;
	static const int vk[8] = { VK_F6, VK_F7, VK_F8, VK_END, VK_HOME, VK_INSERT, VK_F9, VK_F10 };
	int front = game_in_front();
	for (int i = 0; i < 8; i++) {
		int down = front && (GetAsyncKeyState(vk[i]) & 0x8000) != 0;
		if (down && !s->key[i]) {
			if (i == 0) {
				if (cfg.blind) {
					static unsigned rng = 12345u;
					rng = (rng * 1664525u + 1013904223u) ^ (unsigned)GetTickCount64();
					cfg.mode = (int)((rng >> 16) % 3);
					s->trial++;
					say("sim: blind trial %d: %s (hidden), delay %.0f ms\n", s->trial, mode_name[cfg.mode], cfg.latency_ms);
				} else {
					cfg.mode = (cfg.mode + 1) % 3;
					say("sim: mode %s, %.0f ms\n", mode_name[cfg.mode], cfg.latency_ms);
				}
			} else if (i == 1) {
				int n = (int)(sizeof latencies / sizeof latencies[0]), at = 0;
				for (int j = 0; j < n; j++)
					if (fabsf(latencies[j] - cfg.latency_ms) < 0.5f) at = (j + 1) % n;
				cfg.latency_ms = latencies[at];
				say("sim: delay %.0f ms (mode %s)\n", cfg.latency_ms, cfg.blind ? "hidden" : mode_name[cfg.mode]);
			} else if (i == 2) {
				cfg.blind = !cfg.blind;
				say("sim: blind %s\n", cfg.blind ? "on" : "off");
			} else if (i == 3) {
				start_recording(s);
			} else if (i == 4) {
				snapshot_wanted = 1;
			} else if (i == 5) {
				demo_start(s);
			} else if (cfg.blind && s->trial > 0) {
				/* the player's answer in a blind trial: F9 no lag, F10 lag (doom/blind_report.py) */
				say("sim: blind trial %d guess: %s (truth %s, delay %.0f ms)\n", s->trial, i == 6 ? "no-lag" : "lag",
				    mode_name[cfg.mode], cfg.latency_ms);
			}
		}
		s->key[i] = down;
	}
}

/* the depth image with the most draws this frame, of the screen's shape; its layout */
static img_t *pick_depth(sim_t *s, VkImageLayout *layout)
{
	img_t *best = NULL;
	uint64_t most = 0, kept = 0;
	double aspect = (double)s->W / s->H;
	AcquireSRWLockExclusive(&sub_lock);
	AcquireSRWLockShared(&obj_lock);
	for (size_t i = 0; i < images.cap; i++) {
		if (images.st[i] != 1) continue;
		img_t *img = images.v[i];
		uint64_t n = img->draws;
		img->draws = 0;
		if (!n || img->dead || !(img->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) continue;
		double a = (double)img->extent.width / (img->extent.height ? img->extent.height : 1);
		VkDeviceSize bytes = (VkDeviceSize)img->extent.width * img->extent.height * (depth_kind(img->format) == 4 ? 2 : 4);
		if (fabs(a - aspect) > 0.03 * aspect || img->extent.width < s->W / 4 || bytes > s->depth_cap) continue;
		if (img == s->dimg) kept = n;
		if (n > most) most = n, best = img;
	}
	/* Stay with the image followed while it is drawn into: in DOOM two of the screen's size get
	 * about as many draws a frame and take turns leading; a switch is taken only after the other
	 * one has led for 30 frames in a row. */
	if (best && s->dimg && best != s->dimg && kept) {
		if (++s->dpending < 30) best = s->dimg;
		else s->dpending = 0;
	} else {
		s->dpending = 0;
	}
	*layout = best ? best->layout : VK_IMAGE_LAYOUT_UNDEFINED;
	ReleaseSRWLockShared(&obj_lock);
	ReleaseSRWLockExclusive(&sub_lock);
	if (best != s->dimg) {
		if (best)
			say("sim: depth buffer %ux%u, format %d, layout %d, store op %d, %s, %llu draws this frame\n", best->extent.width,
			    best->extent.height, (int)best->format, (int)*layout, (int)best->store,
			    best->sharing == VK_SHARING_MODE_CONCURRENT ? "concurrent" : "exclusive", (unsigned long long)most);
		else if (s->dimg)
			say("sim: no depth buffer this frame\n");
		s->dimg = best;
	}
	return best;
}

static uint32_t lock_key(float t, uint32_t flags)
{
	if (!(t > 0.0f)) return 0;
	int rev = (flags & 2) != 0;
	if (t > 1.0f) t = 1.0f;
	if (flags & 4) {
		uint32_t v = (uint32_t)lrintf(t * 65535.0f);
		return rev ? 0xFFFFu - v : v;
	}
	if (flags & 1) {
		uint32_t v = (uint32_t)lrintf(t * 16777215.0f);
		return rev ? 0xFFFFFFu - v : v;
	}
	float x = rev ? 1.0f - t : t;
	uint32_t u;
	memcpy(&u, &x, 4);
	return u;
}

static void cbar(device_t *d, VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags sa, VkPipelineStageFlags dst, VkAccessFlags da)
{
	VkMemoryBarrier m = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, sa, da };
	d->CmdPipelineBarrier(cb, src, dst, 0, 1, &m, 0, NULL, 0, NULL);
}

static void run(device_t *d, sim_t *s, VkCommandBuffer cb, int k, VkDescriptorSet set, uint32_t gx, uint32_t gy, uint32_t a,
		uint32_t b, uint32_t c, uint32_t e)
{
	uint32_t pc[4] = { a, b, c, e };
	d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipe[k]);
	d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl, 0, 1, &set, 0, NULL);
	d->CmdPushConstants(cb, s->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
	d->CmdDispatch(cb, gx ? gx : 1, gy ? gy : 1, 1);
	cbar(d, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
	if (s->prof_img >= 0 && s->prof_n[s->prof_img] < PROF_MAX) {
		int i = s->prof_img, n = s->prof_n[i]++;
		d->CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s->qpool, s->ts_stride * (uint32_t)i + 4 + (uint32_t)n);
		s->prof_k[i][n] = (unsigned char)k;
	}
}

static VkSemaphore sim_frame(device_t *d, sim_t *s, VkQueue queue, const VkPresentInfoKHR *info, sc_t *sc)
{
	uint32_t i = info->pImageIndices[0];
	if (i >= s->nimg || queue != s->queue) {
		AcquireSRWLockShared(&obj_lock);
		uint32_t fam = (uint32_t)(uintptr_t)map_get(&queues, H64(queue));
		ReleaseSRWLockShared(&obj_lock);
		if (i >= s->nimg || fam != s->family + 1) {
			s->skip = i >= s->nimg ? SKIP_IMAGE : fam ? SKIP_FAMILY : SKIP_QUEUE;
			s->skip_queue = queue, s->skip_fam = fam;
			return VK_NULL_HANDLE;
		}
	}
	poll(s);
	if (s->used[i]) {
		VkResult w = d->WaitForFences(d->handle, 1, &s->fence[i], VK_TRUE, 2000000000ull);
		if (w != VK_SUCCESS) {
			say("sim: our work on image %u did not finish (%d); the simulator stops\n", i, (int)w);
			s->failed = 1;
			return VK_NULL_HANDLE;
		}
		collect(d, s, i);
		if (s->ts_at[i]) {
			uint64_t t4[4] = { 0, 0, 0, 0 };
			if (d->GetQueryPoolResults(d->handle, s->qpool, s->ts_stride * i, 4, sizeof t4, t4, sizeof t4[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
				uint64_t mask = s->ts_bits >= 64 ? ~0ULL : ((1ULL << s->ts_bits) - 1);
				double per = d->props.limits.timestampPeriod * 1e-6;
				double ms = (double)((t4[3] - t4[0]) & mask) * per;
				if (ms >= 0.0 && ms < 1000.0) {
					s->ts_sum_ms += ms, s->ts_n++;
					if (ms > s->ts_max_ms) s->ts_max_ms = ms;
					s->ts_part[0] += (double)((t4[1] - t4[0]) & mask) * per;   /* the game's frame and depth into the ring */
					s->ts_part[1] += (double)((t4[2] - t4[1]) & mask) * per;   /* the warp, the fill, the zoom */
					s->ts_part[2] += (double)((t4[3] - t4[2]) & mask) * per;   /* overlay, recordings, back to the screen */
					int pn = s->prof_n[i];
					uint64_t tp[PROF_MAX];
					if (pn > 0 && d->GetQueryPoolResults(d->handle, s->qpool, s->ts_stride * i + 4, (uint32_t)pn, (size_t)pn * sizeof tp[0],
									     tp, sizeof tp[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
						int occ[K_COUNT] = { 0 };
						uint64_t prev = t4[1];
						for (int n = 0; n < pn; n++) {
							int kk = s->prof_k[i][n], o = occ[kk] < PROF_OCC ? occ[kk] : PROF_OCC;
							occ[kk]++;
							s->prof_ms[kk][o] += (double)((tp[n] - prev) & mask) * per;
							s->prof_cnt[kk]++;
							prev = tp[n];
						}
						s->prof_frames++;
					}
				}
			}
			s->prof_n[i] = 0;
			s->ts_at[i] = 0;
		}
		s->used[i] = 0;
	}
	d->ResetFences(d->handle, 1, &s->fence[i]);
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	uint32_t k = (uint32_t)(s->frames % s->slots);

	/* the game's depth: copied by our command buffer on the present queue, or, on a family without
	 * graphics, first on the game's graphics queue (gcopy; q_lock keeps the game's submits from coming
	 * between the layout read here and that copy) */
	VkImageLayout dl = VK_IMAGE_LAYOUT_UNDEFINED;
	int via_g = !s->graphics && s->gqueue && cfg.depth;
	if (via_g) AcquireSRWLockExclusive(&q_lock);
	img_t *dimg = cfg.depth ? pick_depth(s, &dl) : NULL;
	if (!s->graphics && !s->gqueue) dimg = NULL;   /* no graphics here and no copy through the game's (sim_init) */
	int use_depth = dimg && dl != VK_IMAGE_LAYOUT_UNDEFINED && dl != VK_IMAGE_LAYOUT_PREINITIALIZED;
	if (via_g) {
		if (use_depth && !gcopy(d, s, i, dimg, dl)) {
			use_depth = 0;
			static int said;
			if (!said) said = 1, say("sim: the depth copy on the game's graphics queue failed; this frame goes without depth\n");
		}
		ReleaseSRWLockExclusive(&q_lock);
	}
	int gwait = use_depth && !s->graphics;   /* our work waits for gsem[i] and takes the depth from gstage[i] */
	uint32_t rev = cfg.reversed >= 0 ? (uint32_t)cfg.reversed : (s->reversed > 0 ? 1u : 0u);
	s->dok[k] = use_depth;
	s->drw[k] = use_depth ? dimg->extent.width : 0;
	s->drh[k] = use_depth ? dimg->extent.height : 0;
	s->dfl[k] = use_depth ? depth_kind(dimg->format) | (rev ? 2u : 0u) : 0u;
	s->keys[k] = sample_keys(s);

	/* The network for this frame: its delay, sim_latency plus sim_jitter_ms of a correlated random
	 * walk (AR(1), 0.9 a frame, unit spread: a queue that fills and drains), or lost (sim_loss).
	 * With neither, every frame arrives exactly `latency` after it was made. */
	{
		s->rng = s->rng * 6364136223846793005ULL + 1442695040888963407ULL;
		double u1 = ((double)(s->rng >> 11) + 0.5) / 9007199254740992.0;
		s->rng = s->rng * 6364136223846793005ULL + 1442695040888963407ULL;
		double u2 = ((double)(s->rng >> 11) + 0.5) / 9007199254740992.0;
		double g = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
		s->jit = 0.9 * s->jit + 0.4358899 * g;
		double lat = cfg.latency_ms + cfg.jitter_ms * s->jit;
		if (lat < 1.0) lat = 1.0;
		s->arrive[k] = now.QuadPart + (LONGLONG)(lat * 1e-3 * (double)qpf.QuadPart);
		s->rng = s->rng * 6364136223846793005ULL + 1442695040888963407ULL;
		s->lost[k] = cfg.loss > 0.0f && ((double)(s->rng >> 11) / 9007199254740992.0) < cfg.loss;
		if (s->lost[k]) s->lost_count++;
	}
	/* the frame the cloud player has: the newest that has arrived (slot k is this frame's) */
	unsigned long long have = s->frames < s->slots - 1 ? s->frames : s->slots - 1, jback = 0;
	int j = -1;
	for (unsigned long long back = 0; back < have; back++) {
		int sl = (int)((s->frames - 1 - back) % s->slots);
		if (!s->lost[sl] && now.QuadPart >= s->arrive[sl]) {
			j = sl;
			jback = back;
			break;
		}
	}
	if (j < 0 && have == s->slots - 1 && have) {
		j = (int)((s->frames - have) % s->slots);
		jback = have - 1;
		if (!s->short_ring_said) {
			s->short_ring_said = 1;
			say("sim: the ring of %u frames is shorter than %.0f ms at this frame rate; the oldest is shown\n", s->slots,
			    cfg.latency_ms);
		}
	}
	int mode = cfg.mode;
	uint32_t src = k;
	if (mode != SIM_LOCAL) {
		if (j >= 0) src = (uint32_t)j;
		else mode = SIM_LOCAL;           /* the ring is still filling */
	}
	s->qpc[k] = now.QuadPart;

	double R[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, mx = 0.0, my = 0.0, wv[3] = { 0, 0, 0 };
	if (mode == SIM_WARP) {
		LONGLONG tau = (LONGLONG)(cfg.tau_ms * 1e-3 * (double)qpf.QuadPart);
		mouse_counts(s, s->qpc[src] - tau, now.QuadPart - tau, &mx, &my);   /* for the dump's line */
		predict_turn(s, s->qpc[src], now.QuadPart, wv);                     /* the mouse and the stick */
		rodrigues(wv, R);
	}
	/* the move as well (sim_move): the splat path, which needs the old frame's depth */
	double T[3] = { 0, 0, 0 };
	int splat = mode == SIM_WARP && cfg.move && cfg.move_speed > 0.0f && s->dok[src];
	if (splat) {
		LONGLONG mt = (LONGLONG)((cfg.move_tau_ms >= 0 ? cfg.move_tau_ms : cfg.tau_ms) * 1e-3 * (double)qpf.QuadPart);
		move_over(s, s->qpc[src] - mt, now.QuadPart - mt, T);
	}
	memcpy(s->last_T, T, sizeof T);
	struct params par;
	memset(&par, 0, sizeof par);
	par.T[0] = (float)T[0], par.T[1] = (float)T[1], par.T[2] = (float)T[2], par.T[3] = splat ? 1.0f : 0.0f;
	/* older frames for what the old frame turned away from (sim_history): 4 and 10 frames before it
	 * and the oldest the ring holds (most often the one that saw it), each turned to now by the
	 * mouse the same way; the first that has a pixel fills it */
	int nh = 0;
	uint32_t hslot[3];
	if (mode == SIM_WARP && !splat && cfg.history && have) {
		unsigned long long cand[3] = { jback + 4, jback + 10, have - 1 };
		for (int hh = 0; hh < 3; hh++) {
			unsigned long long b2 = cand[hh];
			if (b2 >= have || b2 <= jback || (hh && b2 <= cand[hh - 1] && hh < 2) || (hh == 2 && (b2 == cand[0] || b2 == cand[1])))
				continue;
			uint32_t sl = (uint32_t)((s->frames - 1 - b2) % s->slots);
			if (s->lost[sl] || now.QuadPart < s->arrive[sl]) continue;
			double Rh[3][3], wh[3];
			predict_turn(s, s->qpc[sl], now.QuadPart, wh);
			rodrigues(wh, Rh);
			for (int r = 0; r < 3; r++)
				for (int c = 0; c < 3; c++) par.HC[nh * 3 + r][c] = (float)Rh[c][r];
			hslot[nh++] = sl;
		}
	}
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++) par.C[r][c] = (float)R[c][r];
	par.f = (float)(s->W * 0.5 / tan(cfg.hfov * 3.14159265358979 / 360.0));
	par.cx = s->W * 0.5f, par.cy = s->H * 0.5f, par.keep = cfg.keep;
	par.W = s->W, par.H = s->H, par.RW = s->drw[src] ? s->drw[src] : 1, par.RH = s->drh[src] ? s->drh[src] : 1;
	par.mark = cfg.blind || mode == SIM_LOCAL ? 0 : (mode == SIM_WARP ? 2 : 1);
	par.dflags = s->dfl[src];
	par.lockz = s->dok[src] ? lock_key(cfg.weapon, s->dfl[src]) : 0;
	par.bgra = s->bgra;
	par.hole = (uint32_t)lrintf(cfg.hole * s->H);
	par.cross = (uint32_t)lrintf(cfg.cross * s->H);
	par.nlock = 4;
	par.depth_ok = (uint32_t)s->dok[src];
	for (int r = 0; r < 4; r++) {
		const float *q = cfg.lock[r];
		if (!(q[2] > q[0] && q[3] > q[1])) continue;
		par.lock[r][0] = (uint32_t)lrintf(q[0] * s->W), par.lock[r][1] = (uint32_t)lrintf(q[1] * s->H);
		par.lock[r][2] = (uint32_t)lrintf(q[2] * s->W), par.lock[r][3] = (uint32_t)lrintf(q[3] * s->H);
	}

	VkCommandBuffer cb = s->cmd[i];
	VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
	if (d->BeginCommandBuffer(cb, &bi) != VK_SUCCESS) {
		s->skip = SKIP_CMDBUF;
		return VK_NULL_HANDLE;
	}
	if (s->ts_ok) {
		d->CmdResetQueryPool(cb, s->qpool, s->ts_stride * i, s->ts_stride);
		d->CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, s->qpool, s->ts_stride * i);
		s->prof_n[i] = 0;
		s->prof_img = cfg.profile ? (int)i : -1;
	}
	const VkPipelineStageFlags CT = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
	cbar(d, cb, CT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, CT,
	     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
	d->CmdUpdateBuffer(cb, s->params.b, 0, sizeof par, &par);

	/* the game's frame and depth into the ring */
	VkImage image = sc->images[i];
	VkImageMemoryBarrier ib[2];
	uint32_t nb = 0;
	ib[nb++] = (VkImageMemoryBarrier){ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_READ_BIT,
					   VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
					   VK_QUEUE_FAMILY_IGNORED, image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
	if (use_depth && !gwait)
		ib[nb++] = (VkImageMemoryBarrier){ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL,
						   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
						   VK_ACCESS_TRANSFER_READ_BIT, dl, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
						   VK_QUEUE_FAMILY_IGNORED, dimg->image, { dimg->aspect, 0, 1, 0, 1 } };
	d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, nb, ib);
	VkBufferImageCopy rc = { 0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { 0, 0, 0 }, { s->W, s->H, 1 } };
	d->CmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s->color[k].b, 1, &rc);
	if (gwait) {                             /* copied on the game's graphics queue; gsem[i] waited at transfer */
		VkBufferCopy bc = { 0, 0, s->gbytes[i] };
		d->CmdCopyBuffer(cb, s->gstage[i].b, s->depth[k].b, 1, &bc);
	} else if (use_depth) {
		VkBufferImageCopy dc = { 0, 0, 0, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 }, { 0, 0, 0 }, { s->drw[k], s->drh[k], 1 } };
		d->CmdCopyImageToBuffer(cb, dimg->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s->depth[k].b, 1, &dc);
		VkImageMemoryBarrier back = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_READ_BIT,
					      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dl,
					      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, dimg->image, { dimg->aspect, 0, 1, 0, 1 } };
		d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &back);
	}
	cbar(d, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT);   /* the params too */

	/* the screen */
	if (s->ts_ok) d->CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, s->qpool, s->ts_stride * i + 1);
	uint32_t gx = (s->W + 7) / 8, gy = (s->H + 7) / 8;
	if (splat) {
		run(d, s, cb, K_CLEAR, s->set_frame[src], gx, gy, 0, 0, 0, 0);
		run(d, s, cb, K_SPLATZ, s->set_frame[src], gx, gy, 0, 0, 0, 0);
		run(d, s, cb, K_SPLATC, s->set_frame[src], gx, gy, 0, 0, 0, 0);
	} else {
		run(d, s, cb, K_BWARP, s->set_frame[src], gx, gy, 0, 0, 0, 0);
		for (int hh = 0; hh < nh; hh++) run(d, s, cb, K_HFILL, s->set_frame[hslot[hh]], gx, gy, (uint32_t)hh, 0, 0, 0);
	}
	for (uint32_t l = 0; l < s->levels; l++)
		run(d, s, cb, K_PDOWN, s->set_level[l], (s->lw[l + 1] + 7) / 8, (s->lh[l + 1] + 7) / 8, s->lw[l], s->lh[l], s->lw[l + 1], s->lh[l + 1]);
	if (s->levels) {
		uint32_t l = s->levels - 1;
		run(d, s, cb, K_PFILL, s->set_level[l], 1, 1, s->lw[l], s->lh[l], s->lw[l + 1], s->lh[l + 1]);
	}
	for (int l = (int)s->levels - 1; l >= 0; l--)
		run(d, s, cb, K_PUP, s->set_level[l], (s->lw[l] + 7) / 8, (s->lh[l] + 7) / 8, s->lw[l], s->lh[l], s->lw[l + 1], s->lh[l + 1]);
	run(d, s, cb, K_ZOOM, s->set_zoom, gx, gy, 0, 0, 0, 0);
	if (s->ts_ok) d->CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s->qpool, s->ts_stride * i + 2);
	if (par.cross) {
		uint32_t half = par.cross + 2, box = 2 * half;
		run(d, s, cb, K_OVERLAY, s->set_zoom, (box + 7) / 8, (box + 7) / 8, s->W / 2 - half, s->H / 2 - half, 1, box);
	}
	if (par.mark) run(d, s, cb, K_OVERLAY, s->set_zoom, 3, 3, 12, 12, 2, 24);
	uint32_t base = i * s->rec_stride;
	if (s->rec_left > 0) {
		run(d, s, cb, K_LUMA, s->set_frame[k], (s->wpr + 7) / 8, (s->lumh + 7) / 8, base, 0, 0, 0);
		s->rec_at[i] = 1, s->rec_q[i] = now.QuadPart, s->rec_m[i] = (cfg.blind ? 3 + mode : mode) | (int)(s->keys[k] << 8);
		s->rec_left--;
	}
	if (s->demo_left > 0) {
		uint32_t hw = s->W / 2, hh = s->H / 2;
		run(d, s, cb, K_HALF, s->set_frame[k], (hw / 2 + 7) / 8, (hh + 7) / 8, base + s->demo_off,
		    (uint32_t)s->dok[k] | (s->dfl[k] << 8), s->drw[k] | (s->drh[k] << 16), base + s->demo_doff);
		s->demo_at[i] = 1, s->demo_q[i] = now.QuadPart, s->demo_m[i] = (cfg.blind ? 3 + mode : mode) | (int)(s->keys[k] << 8);
		s->demo_left--;
	}
	if (s->frames >= s->next_stat && s->dok[k]) {
		s->next_stat = s->frames + 90;
		run(d, s, cb, K_DSTAT, s->set_frame[k], 8, 5, base + s->luma_words, 1, s->drw[k] | (s->drh[k] << 16), s->dfl[k]);
		s->stat_at[i] = 1, s->stat_fl[i] = s->dfl[k];
	}
	cbar(d, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
	     VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
	if (!s->snap_busy && (snapshot_wanted || dump_wanted)) {
		VkDeviceSize frame = (VkDeviceSize)s->W * s->H * 4;
		VkBufferCopy a = { 0, 0, frame }, b = { 0, frame, frame };
		int len = 0;
		cbar(d, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		     VK_ACCESS_TRANSFER_READ_BIT);   /* the ring's copy at the top, read again */
		if (dump_wanted) {
			dump_wanted = 0;
			s->snap_kind = 2;
			d->CmdCopyBuffer(cb, s->fin.b, s->snap.b, 1, &a);
			d->CmdCopyBuffer(cb, s->color[src].b, s->snap.b, 1, &b);
			s->snap_bytes0 = s->snap_bytes1 = (uint32_t)frame;
			len = snprintf(s->snap_info, sizeof s->snap_info,
				       "frame %llu\nmode %s\nblind %d\nsrc_age_ms %.3f\nmouse %.3f %.3f\nscreen %u %u\nbgra %u\nf %.4f\nkeep %.4f\n"
				       "R %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\ndepth_ok %d\ndepth %u %u flags %u\nlockz 0x%08X\n"
				       "T %.6f %.6f %.6f\nsplat %d\nkeys %u\nw %.8f %.8f %.8f\n",
				       s->frames, mode_name[mode], cfg.blind, (now.QuadPart - s->qpc[src]) * 1e3 / qpf.QuadPart, mx, my, s->W,
				       s->H, s->bgra, par.f, par.keep, R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1],
				       R[2][2], s->dok[src], s->drw[src], s->drh[src], s->dfl[src], par.lockz, T[0], T[1], T[2], splat,
				       s->keys[k], wv[0], wv[1], wv[2]);
		} else {
			snapshot_wanted = 0;
			s->snap_kind = 1;
			VkDeviceSize dbytes = (VkDeviceSize)s->drw[k] * s->drh[k] * ((s->dfl[k] & 4) ? 2 : 4);
			d->CmdCopyBuffer(cb, s->color[k].b, s->snap.b, 1, &a);
			if (s->dok[k] && dbytes) {
				b.size = dbytes;
				d->CmdCopyBuffer(cb, s->depth[k].b, s->snap.b, 1, &b);
			}
			s->snap_bytes0 = (uint32_t)frame, s->snap_bytes1 = s->dok[k] ? (uint32_t)dbytes : 0;
			len = snprintf(s->snap_info, sizeof s->snap_info,
				       "frame %llu\nscreen %u %u\nformat %d\nbgra %u\ndepth_ok %d\ndepth %u %u\ndepth_format %d\ndepth_flags %u\n"
				       "depth_layout %d\ndepth_store_op %d\nhfov %.4f\n",
				       s->frames, s->W, s->H, (int)sc->format, s->bgra, s->dok[k], s->drw[k], s->drh[k],
				       dimg ? (int)dimg->format : -1, s->dfl[k], (int)dl, dimg ? (int)dimg->store : -1, cfg.hfov);
		}
		(void)len;
		s->snap_at[i] = 1, s->snap_busy = 1;
		cbar(d, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
	}
	VkImageMemoryBarrier to_dst = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED,
					VK_QUEUE_FAMILY_IGNORED, image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
	d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to_dst);
	d->CmdCopyBufferToImage(cb, s->fin.b, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rc);
	VkImageMemoryBarrier to_present = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
					    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_QUEUE_FAMILY_IGNORED,
					    VK_QUEUE_FAMILY_IGNORED, image, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
	d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &to_present);
	if (s->ts_ok) {
		d->CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s->qpool, s->ts_stride * i + 3);
		s->ts_at[i] = 1;
	}
	s->prof_img = -1;
	if (d->EndCommandBuffer(cb) != VK_SUCCESS) {
		s->skip = SKIP_CMDBUF;
		return VK_NULL_HANDLE;
	}

	VkSemaphore wsem[17];
	VkPipelineStageFlags waits[17];
	uint32_t nw = info->waitSemaphoreCount < 16 ? info->waitSemaphoreCount : 16;
	for (uint32_t w = 0; w < nw; w++) wsem[w] = info->pWaitSemaphores[w], waits[w] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	if (gwait) wsem[nw] = s->gsem[i], waits[nw++] = VK_PIPELINE_STAGE_TRANSFER_BIT;   /* the depth from gcopy */
	VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, nw, wsem, waits, 1, &cb, 1, &s->sem[i] };
	AcquireSRWLockExclusive(&q_lock);
	VkResult r = d->QueueSubmit(queue, 1, &si, s->fence[i]);
	ReleaseSRWLockExclusive(&q_lock);
	if (r == VK_SUCCESS && gwait) s->gsig[i] = 0;
	if (r != VK_SUCCESS) {
		say("sim: our submit failed (%d); the simulator stops\n", (int)r);
		s->failed = 1;
		s->rec_at[i] = s->stat_at[i] = s->snap_at[i] = 0;
		return VK_NULL_HANDLE;
	}
	s->used[i] = 1;
	s->frames++;

	/* a line every 5 s */
	ULONGLONG tick = GetTickCount64();
	if (tick >= s->next_log) {
		if (s->next_log) {
			double secs = (now.QuadPart - s->log_q) / (double)qpf.QuadPart;
			int64_t mice = s->shm ? s->shm->count : 0;
			int64_t padn = s->shm && s->shm->version >= 2 ? s->shm->pad_count : 0;
			if (padn != s->log_pads || (s->shm && s->shm->version >= 2 && s->shm->pad_connected != s->log_pad_on))
				say("sim: gamepad %s, %.0f samples/s%s\n", s->shm->pad_connected ? "connected" : "not connected",
				    (padn - s->log_pads) / (secs > 0 ? secs : 1), cfg.pad ? " (sim_pad=1: it turns the warp)" : " (sim_pad=0)");
			s->log_pads = padn, s->log_pad_on = s->shm && s->shm->version >= 2 ? s->shm->pad_connected : 0;
			say("sim: %.1f fps, mode %s%s, delay %.0f ms (frame %.1f ms old), depth %s, mouse %.0f events/s, %llu frames lost, "
			    "move %.3f %.3f; our GPU time %.2f ms a frame (max %.2f: into the ring %.2f, warp + fill + zoom %.2f, the rest %.2f)\n",
			    (s->frames - s->log_frames) / (secs > 0 ? secs : 1), mode_name[mode], cfg.blind ? " (blind)" : "", cfg.latency_ms,
			    (now.QuadPart - s->qpc[src]) * 1e3 / qpf.QuadPart, use_depth ? "yes" : "no",
			    (mice - s->log_mice) / (secs > 0 ? secs : 1), s->lost_count, s->last_T[0], s->last_T[2],
			    s->ts_n ? s->ts_sum_ms / s->ts_n : -1.0, s->ts_max_ms, s->ts_n ? s->ts_part[0] / s->ts_n : -1.0,
			    s->ts_n ? s->ts_part[1] / s->ts_n : -1.0, s->ts_n ? s->ts_part[2] / s->ts_n : -1.0);
			s->ts_sum_ms = s->ts_max_ms = 0.0, s->ts_n = 0;
			s->ts_part[0] = s->ts_part[1] = s->ts_part[2] = 0.0;
			if (s->prof_frames) {
				char line[1536];
				size_t at = 0;
				double all = 0.0;
				for (int kk = 0; kk < K_COUNT && at < sizeof line - 160; kk++) {
					if (!s->prof_cnt[kk]) continue;
					double sum = 0.0;
					for (int o = 0; o <= PROF_OCC; o++) sum += s->prof_ms[kk][o];
					sum /= s->prof_frames, all += sum;
					double per_frame = (double)s->prof_cnt[kk] / s->prof_frames;
					at += (size_t)snprintf(line + at, sizeof line - at, "%s%s %.3f", at ? ", " : "", kernel_name[kk], sum);
					if (per_frame > 1.01) {
						at += (size_t)snprintf(line + at, sizeof line - at, " (%.1f a frame:", per_frame);
						for (int o = 0; o <= PROF_OCC && at < sizeof line - 16; o++)
							if (s->prof_ms[kk][o] > 0.0) at += (size_t)snprintf(line + at, sizeof line - at, " %.3f", s->prof_ms[kk][o] / s->prof_frames);
						at += (size_t)snprintf(line + at, sizeof line - at, ")");
					}
				}
				say("sim: profile, ms a frame, each dispatch with its barrier, %u frames: %s; all %.3f\n", s->prof_frames,
				    at ? line : "-", all);
				memset(s->prof_ms, 0, sizeof s->prof_ms);
				memset(s->prof_cnt, 0, sizeof s->prof_cnt);
				s->prof_frames = 0;
			}
			s->log_mice = mice;
		}
		s->next_log = tick + 5000, s->log_frames = s->frames, s->log_q = now.QuadPart;
	}
	return s->sem[i];
}

/* A present the simulator did not take (why > 0) or took (why 0): the log says when they start to
 * pass it by, why, every 5 s while they do, and when they reach it again. */
static void bypass_note(sim_t *s, int why, VkQueue queue)
{
	static LONG64 counts[SKIP_N];
	static ULONGLONG next;
	static int passing;
	if (!why) {
		if (passing) say("sim: the game's frames reach the simulator again\n");
		passing = 0;
		return;
	}
	counts[why]++;
	ULONGLONG now = GetTickCount64();
	if (!passing) {
		passing = 1, next = now + 5000;
		char fam[24] = "never seen";
		if (why == SKIP_FAMILY) snprintf(fam, sizeof fam, "%u", s->skip_fam - 1);
		say("sim: the game's frames pass the simulator by: %s (queue %p of family %s; the simulator's queue %p, family %u)."
		    " F6, End and the rest do nothing while this lasts\n", skip_name[why], (void *)queue,
		    why == SKIP_FAMILY || why == SKIP_QUEUE ? fam : "-", (void *)s->queue, s->family);
	} else if (now >= next) {
		char line[512];
		int at = 0;
		for (int k = 1; k < SKIP_N && at < (int)sizeof line - 96; k++)
			if (counts[k]) at += snprintf(line + at, sizeof line - (size_t)at, "%s%s %lld", at ? ", " : "", skip_name[k], counts[k]);
		say("sim: frames passing the simulator by in the last 5 s: %s\n", at ? line : "-");
		memset(counts, 0, sizeof counts);
		next = now + 5000;
	}
}

static VKAPI_ATTR VkResult VKAPI_CALL h_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *info)
{
	device_t *d = dev_of(queue);
	sim_t *s = &d->sim;
	VkSemaphore ours = VK_NULL_HANDLE;
	int why = SKIP_NONE;
	if (cfg.on && !s->failed && info->swapchainCount == 1) {
		AcquireSRWLockShared(&obj_lock);
		sc_t *sc = map_get(&swapchains, H64(info->pSwapchains[0]));
		ReleaseSRWLockShared(&obj_lock);
		if (sc && sc != s->refused) {
			if (s->ready && s->sc != sc) sim_release(d, s, 0);
			if (s->ready && queue != s->queue) {
				/* The game presents from a queue of another family now (DOOM can, from its compute queue:
				 * every family here but video decode can present). Our work must go to the queue that
				 * presents, so the simulator starts again there; at most once in 2 s, so a game that
				 * keeps switching is passed by instead (and the log says so). */
				AcquireSRWLockShared(&obj_lock);
				uint32_t fam = (uint32_t)(uintptr_t)map_get(&queues, H64(queue));
				ReleaseSRWLockShared(&obj_lock);
				ULONGLONG tick = GetTickCount64();
				if (fam && fam - 1 != s->family && tick >= s->move_ok) {
					say("sim: the game presents from a queue of family %u now (the simulator was on family %u): it starts "
					    "again there\n", fam - 1, s->family);
					sim_release(d, s, 2);
					s->move_ok = tick + 2000;
				}
			}
			if (!s->ready && !sim_init(d, s, queue, sc)) {
				sim_release(d, s, 0);
				s->refused = sc;
			}
			if (s->ready) {
				s->skip = SKIP_NONE;
				ours = sim_frame(d, s, queue, info, sc);
				if (!ours) why = s->skip ? s->skip : SKIP_OTHER;
			} else {
				why = SKIP_REFUSED;
			}
		} else {
			why = sc ? SKIP_REFUSED : SKIP_SWAPCHAIN;
		}
	} else if (cfg.on) {
		why = s->failed ? SKIP_FAILED : SKIP_MANY;
	}
	if (cfg.on) bypass_note(s, why, queue);
	if (!ours) return d->QueuePresentKHR(queue, info);
	VkPresentInfoKHR p = *info;
	p.waitSemaphoreCount = 1;
	p.pWaitSemaphores = &ours;
	return d->QueuePresentKHR(queue, &p);
}

/* ---------------------------------------------------------------- the loader's entry points */

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name);

#define HOOK(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)h_##n;
static PFN_vkVoidFunction device_hook(const char *name)
{
	HOOK(DestroyDevice) HOOK(GetDeviceQueue) HOOK(GetDeviceQueue2) HOOK(CreateSwapchainKHR) HOOK(DestroySwapchainKHR)
	HOOK(QueuePresentKHR)
	if (cfg.depth) {
		HOOK(QueueSubmit) HOOK(CreateImage) HOOK(DestroyImage) HOOK(CreateImageView) HOOK(DestroyImageView)
		HOOK(CreateFramebuffer) HOOK(DestroyFramebuffer) HOOK(CreateRenderPass) HOOK(DestroyRenderPass)
		HOOK(BeginCommandBuffer) HOOK(CmdBeginRenderPass) HOOK(CmdEndRenderPass) HOOK(CmdDraw) HOOK(CmdDrawIndexed)
		HOOK(CmdDrawIndirect) HOOK(CmdDrawIndexedIndirect) HOOK(CmdPipelineBarrier) HOOK(CmdExecuteCommands)
	}
	return NULL;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name)
{
	if (!name) return NULL;
	if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
	device_t *d = device ? dev_of(device) : NULL;
	if (!d) return NULL;
	PFN_vkVoidFunction next = d->gdpa(device, name);
	PFN_vkVoidFunction h = next ? device_hook(name) : NULL;
	return h ? h : next;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name)
{
	if (!name) return NULL;
	if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
	if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)h_CreateInstance;
	if (!instance) return NULL;
	if (!strcmp(name, "vkDestroyInstance")) return (PFN_vkVoidFunction)h_DestroyInstance;
	if (!strcmp(name, "vkCreateDevice")) return (PFN_vkVoidFunction)h_CreateDevice;
	if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
	inst_t *in = inst_of(instance);
	if (!in) return NULL;
	PFN_vkVoidFunction next = in->gipa(instance, name);
	PFN_vkVoidFunction h = next ? device_hook(name) : NULL;
	return h ? h : next;
}

VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *v)
{
	if (!v || v->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
	if (v->loaderLayerInterfaceVersion >= 2) {
		v->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
		v->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
		v->pfnGetPhysicalDeviceProcAddr = NULL;
	}
	if (v->loaderLayerInterfaceVersion > 2) v->loaderLayerInterfaceVersion = 2;
	return VK_SUCCESS;
}

BOOL WINAPI DllMain(HINSTANCE self, DWORD reason, LPVOID reserved)
{
	(void)reserved;
	if (reason == DLL_PROCESS_ATTACH) {          /* no DisableThreadLibraryCalls: the static CRT and TLS want them */
		QueryPerformanceFrequency(&qpf);
		QueryPerformanceCounter(&qp0);
		DWORD n = GetModuleFileNameW(self, here, MAX_PATH);
		if (n && n < MAX_PATH) {
			wchar_t *slash = wcsrchr(here, L'\\');
			if (slash) *slash = 0;
		}
	}
	return TRUE;
}
