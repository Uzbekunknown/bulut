/* test_vk_sim.c - the cloud simulator layer (cloud_sim_vk.dll) in a small Vulkan app.
 *
 *   test_vk_sim <build dir> <mode 0|1|2> <latency ms> [validation 0|1]
 *
 * The app draws a panorama around a camera that turns with the mouse (test.frag): a red stripe at
 * world yaw 0, a checker elsewhere at depth 0.9, and a grey box at the bottom right, "the weapon",
 * at depth 0.05. The mouse is the layer's own shared memory: the test writes one event of +100
 * counts into it itself (sim_logger=0), as input_logger would, and turns its own camera by it
 * TAU_MS later, as a game would. Halfway through the delay after the turn shows on the screen, the
 * test has the layer dump what it shows (a named event), and checks:
 *   the stripe  local: where the camera is now; cloud: where it was; warp: where it is now
 *   the weapon  grey in place in every mode; in warp, what it hid is filled, not grey
 *   the crosshair at the centre, the corner mark (red cloud, green warp)
 * Then it asks the layer for a recording and a snapshot through the settings file
 * (sim_record_now, sim_snapshot_now) and checks their files. With validation 1 the Khronos
 * validation layer (with synchronization validation) runs below the simulator, and any error fails.
 *
 * Modes 10 and 11 do the same with the gamepad's right stick (the layer's sim_pad): 10 holds the
 * stick at 0.7 right for 120 ms and checks the warp like mode 2; 11 makes a calibration recording
 * of a camera the stick turns through known rate tables, for fit_pad.py. The test's camera is
 * integrated in small steps, apart from the layer's closed form, so the two check each other.
 *
 * Mode 12 is mode 2 with the frames presented from a compute-only queue family from frame 20 on (the
 * swapchain and the depth shared by both families), as DOOM may present from its compute queue: the
 * layer has to start again on that family and warp as before. */
#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include "mouse_shm.h"
#include "test_spv.h"

#define W 640
#define H 400
#define A_RAD 0.002
#define COUNTS 100
#define TAU_MS 30.0
#define RINGF 2

#define CHECK(x)                                                                              \
	do {                                                                                  \
		VkResult r_ = (x);                                                            \
		if (r_ != VK_SUCCESS) {                                                       \
			printf("FAIL: %s returned %d (line %d)\n", #x, (int)r_, __LINE__);    \
			exit(1);                                                              \
		}                                                                             \
	} while (0)

static int errors, warnings, printed;

static VKAPI_ATTR VkBool32 VKAPI_CALL on_debug(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type,
					       const VkDebugUtilsMessengerCallbackDataEXT *data, void *user)
{
	(void)type, (void)user;
	int error = (sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
	if (!error && !(sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)) return VK_FALSE;
	if (error) errors++;
	else warnings++;
	if (printed++ < 25) printf("validation %s: %s\n\n", error ? "ERROR" : "warning", data->pMessage);
	return VK_FALSE;
}

static LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l) { return DefWindowProcW(h, m, w, l); }

static uint32_t memtype(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want)
{
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(pd, &mp);
	for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
	return 0;
}

/* calib: the camera the fit must find again */
#define CAL_A 0.002          /* yaw, rad a count */
#define CAL_B 0.0015         /* pitch, rad a count */
#define CAL_LAG_MS 8.0
#define CAL_HFOV 75.0
#define CAL_FRAMES 300

static int rec_frames = 30;

/* modes 10 and 11: the right stick turns the camera at a rate, from these tables (rad/s at the
 * stick's distance from the centre 0, 0.1 .. 1; yaw positive to the right, pitch up) through a
 * first-order lag */
static const double PAD_YAW[11] = { 0, 0, 0.15, 0.4, 0.7, 1.0, 1.35, 1.7, 2.1, 2.5, 3.0 };
static const double PAD_PITCH[11] = { 0, 0, 0.1, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 };
#define PAD_LAG_MS 8.0

static double pad_tab(const double *t, double m)
{
	double x = m * 10.0;
	int i = (int)x;
	if (i >= 10) return t[10];
	return t[i] + (t[i + 1] - t[i]) * (x - i);
}

static void pad_write(struct mouse_shm *shm, LONGLONG qpc, double sx, double sy)
{
	struct mouse_shm_pad *e = &shm->pads[shm->pad_count & (MOUSE_SHM_PADS - 1)];
	memset(e, 0, sizeof *e);
	sx = sx < -1.0 ? -1.0 : sx > 1.0 ? 1.0 : sx;
	sy = sy < -1.0 ? -1.0 : sy > 1.0 ? 1.0 : sy;
	e->qpc = qpc;
	e->rx = (int16_t)lround(sx * 32767.0), e->ry = (int16_t)lround(sy * 32767.0);
	e->packet = (uint32_t)shm->pad_count + 1, e->user = 0;
	InterlockedIncrement64((volatile LONG64 *)&shm->pad_count);
}

/* The test's game camera from the stick up to t (QPC): yaw and pitch, the rate held from each
 * sample to the next, through the lag, integrated in 0.05 ms midpoint steps. */
static struct { int64_t next; double t, ry, rp, vy, vp, yaw, pitch; int on; } padcam;

static void pad_camera(const struct mouse_shm *shm, LONGLONG t, double freq, double lag, double *yaw, double *pitch)
{
	double tt = (double)t / freq;
	for (;;) {
		double until = tt;
		int take = 0;
		if (padcam.next < shm->pad_count) {
			double te = (double)shm->pads[padcam.next & (MOUSE_SHM_PADS - 1)].qpc / freq;
			if (te <= tt) until = te, take = 1;
		}
		while (padcam.on && padcam.t < until) {
			double d = until - padcam.t < 5e-5 ? until - padcam.t : 5e-5;
			if (lag > 0.0) {
				double my = padcam.vy + 0.5 * d * (padcam.ry - padcam.vy) / lag;
				double mp = padcam.vp + 0.5 * d * (padcam.rp - padcam.vp) / lag;
				padcam.yaw += d * my, padcam.pitch += d * mp;
				padcam.vy += d * (padcam.ry - my) / lag, padcam.vp += d * (padcam.rp - mp) / lag;
			} else {
				padcam.yaw += d * padcam.ry, padcam.pitch += d * padcam.rp;
			}
			padcam.t += d;
		}
		padcam.t = until;
		if (!take) break;
		const struct mouse_shm_pad *e = &shm->pads[padcam.next & (MOUSE_SHM_PADS - 1)];
		double sx = e->rx / 32767.0, sy = e->ry / 32767.0;
		sx = sx < -1.0 ? -1.0 : sx, sy = sy < -1.0 ? -1.0 : sy;
		double m = sqrt(sx * sx + sy * sy), mm = m > 1.0 ? 1.0 : m;
		padcam.ry = m > 1e-9 ? pad_tab(PAD_YAW, mm) * sx / m : 0.0;
		padcam.rp = m > 1e-9 ? pad_tab(PAD_PITCH, mm) * sy / m : 0.0;
		padcam.on = 1;
		padcam.next++;
	}
	*yaw = padcam.yaw, *pitch = padcam.pitch;
}

/* mode 11's stick: both axes over the whole range, slowly enough for the shift estimator */
static double pad_sx(double t) { return 0.95 * sin(6.2831853 * t / 1.6) * (0.55 + 0.45 * sin(6.2831853 * t / 2.9)); }
static double pad_sy(double t) { return 0.8 * sin(6.2831853 * t / 1.05 + 0.7) * (0.5 + 0.5 * cos(6.2831853 * t / 3.7)); }

static void write_settings(const wchar_t *path, int mode, int latency, const char *extra)
{
	FILE *f = _wfopen(path, L"wb");
	if (!f) {
		printf("FAIL: cannot write the settings\n");
		exit(1);
	}
	fprintf(f, "sim=1\nsim_mode=%d\nsim_latency=%d\nsim_keep=1\nsim_slots=24\nsim_logger=0\nsim_tau_ms=%.1f\nsim_lag_ms=0\n"
		   "sim_hfov=90\nsim_cx=0 %.6f 0\nsim_cy=%.6f 0 0\nsim_weapon_depth=0.2\nsim_depth_reversed=0\nsim_hole=0.02\n"
		   "sim_cross=0.02\nsim_rec_frames=%d\n%s",
		mode, latency, TAU_MS, -A_RAD, A_RAD, rec_frames, extra ? extra : "");
	fclose(f);
}

static double cal_x(double t) { return 150.0 * sin(6.2831853 * t / 1.3) + 60.0 * sin(6.2831853 * t / 0.47 + 1.0); }
static double cal_y(double t) { return 70.0 * sin(6.2831853 * t / 0.9 + 0.5) + 25.0 * sin(6.2831853 * t / 0.31); }

static unsigned char *slurp(const wchar_t *path, size_t *size)
{
	FILE *f = _wfopen(path, L"rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	unsigned char *b = malloc((size_t)n + 1);
	if (b && fread(b, 1, (size_t)n, f) != (size_t)n) n = 0;
	if (b) b[n] = 0;
	fclose(f);
	if (size) *size = (size_t)n;
	return b;
}

static const unsigned char *px(const unsigned char *img, int x, int y) { return img + ((size_t)y * W + x) * 4; }  /* B G R A */

static int near_rgb(const unsigned char *p, int r, int g, int b, int tol)
{
	return abs(p[2] - r) <= tol && abs(p[1] - g) <= tol && abs(p[0] - b) <= tol;
}

int wmain(int argc, wchar_t **argv)
{
	if (argc < 4) {
		printf("usage: test_vk_sim <build dir> <mode 0..11> <latency ms> [validation 0|1]\n"
		       "  0 local, 1 cloud, 2 warp; 3, 4 strafe; 5 bad network; 6, 7 history; 8 walk recording; 9 mouse calibration;\n"
		       "  10 warp by the gamepad's stick; 11 stick calibration recording;\n"
		       "  12 warp with the frames presented from a compute-only queue from frame 20 (the layer must follow)\n");
		return 2;
	}
	const wchar_t *build = argv[1];
	int mode = _wtoi(argv[2]), latency = _wtoi(argv[3]), validation = argc > 4 ? _wtoi(argv[4]) : 0;
	int cq = mode == 12;             /* from frame 20 the frames are presented from a compute-only queue (as DOOM may) */
	if (cq) mode = 2;
	int pad = mode == 10 || mode == 11;  /* the gamepad's right stick turns the camera */
	int padcal = mode == 11;         /* ... for a calibration recording, for fit_pad.py */
	if (mode == 10) mode = 2;
	if (padcal) mode = 9;
	int calib = mode == 9;           /* a recording of a known camera, for fit_doom.py */
	if (calib) mode = 0, rec_frames = CAL_FRAMES;
	int move = mode == 3 || mode == 4;   /* the camera strafes (D) through a cylinder: the splat path, sim_move */
	double move_lag = mode == 4 ? 0.040 : 0.0;   /* 4: the velocity follows the key with a 40 ms lag */
	if (move) mode = 2;
	int walk = mode == 8;            /* a demo recording of a known walk, for fit_move.py */
	if (walk) mode = 0, move = 1;
	int netw = mode == 5;            /* warp over a bad network: 30 ms of jitter, 20 % of the frames lost */
	if (netw) mode = 2;
	int osc = mode == 6 || mode == 7;   /* a turn right and back: the left edge from history (6) or blur (7) */
	int nohist = mode == 7;
	if (osc) mode = 2;
	wchar_t tmp[MAX_PATH], dir[MAX_PATH], settings[MAX_PATH], logpath[MAX_PATH], dump[MAX_PATH], recdir[MAX_PATH];
	GetTempPathW(MAX_PATH, tmp);
	_snwprintf(dir, MAX_PATH, L"%lscloud-sim-vk-test-%lu", tmp, GetCurrentProcessId());
	CreateDirectoryW(dir, NULL);
	_snwprintf(settings, MAX_PATH, L"%ls\\settings.txt", dir);
	_snwprintf(logpath, MAX_PATH, L"%ls\\layer.log", dir);
	_snwprintf(dump, MAX_PATH, L"%ls\\dump", dir);
	_snwprintf(recdir, MAX_PATH, L"%ls\\rec", dir);
	char recdir8[MAX_PATH * 3], extra[MAX_PATH * 3 + 512], padset[400] = "";
	WideCharToMultiByte(CP_UTF8, 0, recdir, -1, recdir8, sizeof recdir8, NULL, NULL);
	if (pad && !padcal) {                /* the layer gets the truth: yaw in its sign (a turn right is -y) */
		int at = snprintf(padset, sizeof padset, "sim_pad=1\nsim_pad_radial=1\nsim_pad_tau_ms=%.1f\nsim_pad_lag_ms=%.1f\nsim_pad_yaw=",
				  TAU_MS, PAD_LAG_MS);
		for (int i = 0; i < 11; i++) at += snprintf(padset + at, sizeof padset - at, "%s%.4f", i ? " " : "", -PAD_YAW[i]);
		at += snprintf(padset + at, sizeof padset - at, "\nsim_pad_pitch=");
		for (int i = 0; i < 11; i++) at += snprintf(padset + at, sizeof padset - at, "%s%.4f", i ? " " : "", PAD_PITCH[i]);
		snprintf(padset + at, sizeof padset - at, "\n");
	}
	snprintf(extra, sizeof extra, "sim_rec_dir=%s\n%s%s%s%s%s%s", recdir8,
		 move ? "sim_move=1\nsim_move_speed=30\n" : "",   /* 3 units a second over a near plane of 0.1 */
		 walk ? "sim_demo_frames=300\n" : "", move_lag > 0 ? "sim_move_lag_ms=40\n" : "",
		 netw ? "sim_jitter_ms=30\nsim_loss=0.2\n" : "", nohist ? "sim_history=0\n" : "", padset);
	write_settings(settings, mode, latency, extra);
	/* the keyboard's stand-in, before the layer's first look for it */
	wchar_t kname[96];
	_snwprintf(kname, 96, L"Local\\cloud-sim-vk-keys-%lu", GetCurrentProcessId());
	HANDLE kmap = move ? CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(LONG), kname) : NULL;
	volatile LONG *kbits = kmap ? MapViewOfFile(kmap, FILE_MAP_WRITE | FILE_MAP_READ, 0, 0, sizeof(LONG)) : NULL;
	if (kbits) *kbits = 0;
	SetEnvironmentVariableW(L"CLOUD_SIM_VK_SETTINGS", settings);
	SetEnvironmentVariableW(L"CLOUD_SIM_VK_LOG", logpath);
	SetEnvironmentVariableW(L"CLOUD_SIM_VK_DUMP", dump);
	SetEnvironmentVariableW(L"VK_ADD_LAYER_PATH", build);
	printf("test_vk_sim: mode %d, delay %d ms, validation %d; files in %ls\n", mode, latency, validation, dir);

	HINSTANCE hi = GetModuleHandleW(NULL);
	WNDCLASSW wc = { 0 };
	wc.lpfnWndProc = wndproc, wc.hInstance = hi, wc.lpszClassName = L"test_vk_sim";
	RegisterClassW(&wc);
	RECT rc = { 0, 0, W, H };
	AdjustWindowRect(&rc, WS_OVERLAPPED | WS_CAPTION, FALSE);
	HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"test_vk_sim", WS_OVERLAPPED | WS_CAPTION, 60, 60, rc.right - rc.left,
				    rc.bottom - rc.top, NULL, NULL, hi, NULL);
	ShowWindow(hwnd, SW_SHOWNOACTIVATE);

	const char *layers[2] = { "VK_LAYER_UZ_cloud_sim", "VK_LAYER_KHRONOS_validation" };
	const char *exts[4] = { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME, VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
				"VK_EXT_validation_features" };
	VkValidationFeatureEnableEXT enables[1] = { VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT };
	VkValidationFeaturesEXT vf = { VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, NULL, 1, enables, 0, NULL };
	VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "test_vk_sim", 1, "none", 1, VK_API_VERSION_1_0 };
	VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, validation ? &vf : NULL, 0, &app, validation ? 2u : 1u, layers,
				     validation ? 4u : 2u, exts };
	VkInstance inst;
	CHECK(vkCreateInstance(&ici, NULL, &inst));
	VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
	if (validation) {
		PFN_vkCreateDebugUtilsMessengerEXT cdm = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
		VkDebugUtilsMessengerCreateInfoEXT mi = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
		mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
				 | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		mi.pfnUserCallback = on_debug;
		if (cdm) CHECK(cdm(inst, &mi, NULL, &messenger));
	}
	VkWin32SurfaceCreateInfoKHR sci = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, NULL, 0, hi, hwnd };
	VkSurfaceKHR surface;
	CHECK(vkCreateWin32SurfaceKHR(inst, &sci, NULL, &surface));
	uint32_t npd = 1;
	VkPhysicalDevice pd;
	vkEnumeratePhysicalDevices(inst, &npd, &pd);
	uint32_t nqf = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
	VkQueueFamilyProperties qfp[16];
	if (nqf > 16) nqf = 16;
	vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp);
	uint32_t family = UINT32_MAX;
	for (uint32_t i = 0; i < nqf && family == UINT32_MAX; i++) {
		VkBool32 present = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
		if ((qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) family = i;
	}
	if (family == UINT32_MAX) {
		printf("FAIL: no graphics queue that presents\n");
		return 1;
	}
	uint32_t cfam = UINT32_MAX;          /* 12: a compute-only family that presents */
	for (uint32_t i = 0; cq && i < nqf && cfam == UINT32_MAX; i++) {
		VkBool32 present = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
		if ((qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) cfam = i;
	}
	if (cq && cfam == UINT32_MAX) {
		printf("SKIPPED: no compute-only queue family here that presents\n");
		return 0;
	}
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci[2] = { { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, family, 1, &prio },
					   { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, cfam, 1, &prio } };
	const char *dext[1] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, cq ? 2u : 1u, qci, 0, NULL, 1, dext, NULL };
	VkDevice dev;
	CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
	VkQueue queue, cqueue = VK_NULL_HANDLE;
	vkGetDeviceQueue(dev, family, 0, &queue);
	if (cq) vkGetDeviceQueue(dev, cfam, 0, &cqueue);
	uint32_t both[2] = { family, cfam };     /* 12: the swapchain and the depth are shared by the two families */

	VkSurfaceCapabilitiesKHR caps;
	CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps));
	uint32_t nfmt = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &nfmt, NULL);
	VkSurfaceFormatKHR fmts[64];
	if (nfmt > 64) nfmt = 64;
	vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &nfmt, fmts);
	VkSurfaceFormatKHR sf = fmts[0];
	for (uint32_t i = 0; i < nfmt; i++)
		if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM) sf = fmts[i];
	if (sf.format != VK_FORMAT_B8G8R8A8_UNORM) {
		printf("FAIL: no B8G8R8A8_UNORM surface format (the checks read BGRA)\n");
		return 1;
	}
	uint32_t nimg = caps.minImageCount + 1;
	if (caps.maxImageCount && nimg > caps.maxImageCount) nimg = caps.maxImageCount;
	VkSwapchainCreateInfoKHR swci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
	swci.surface = surface, swci.minImageCount = nimg, swci.imageFormat = sf.format, swci.imageColorSpace = sf.colorSpace;
	swci.imageExtent = (VkExtent2D){ W, H }, swci.imageArrayLayers = 1, swci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	swci.imageSharingMode = cq ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE, swci.preTransform = caps.currentTransform;
	swci.queueFamilyIndexCount = cq ? 2u : 0u, swci.pQueueFamilyIndices = cq ? both : NULL;
	swci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, swci.presentMode = VK_PRESENT_MODE_FIFO_KHR, swci.clipped = VK_TRUE;
	VkSwapchainKHR swapchain;
	CHECK(vkCreateSwapchainKHR(dev, &swci, NULL, &swapchain));
	VkImage images[8];
	nimg = 8;
	CHECK(vkGetSwapchainImagesKHR(dev, swapchain, &nimg, images));
	VkImageView views[8];
	for (uint32_t i = 0; i < nimg; i++) {
		VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, images[i], VK_IMAGE_VIEW_TYPE_2D, sf.format,
					     { 0 }, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
		CHECK(vkCreateImageView(dev, &vi, NULL, &views[i]));
	}
	VkImageCreateInfo dic = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT, { W, H, 1 }, 1, 1,
				  VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
				  cq ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE, cq ? 2u : 0u, cq ? both : NULL,
				  VK_IMAGE_LAYOUT_UNDEFINED };
	VkImage depth;
	CHECK(vkCreateImage(dev, &dic, NULL, &depth));
	VkMemoryRequirements mr;
	vkGetImageMemoryRequirements(dev, depth, &mr);
	VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size,
				     memtype(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
	VkDeviceMemory dmem;
	CHECK(vkAllocateMemory(dev, &mai, NULL, &dmem));
	CHECK(vkBindImageMemory(dev, depth, dmem, 0));
	VkImageViewCreateInfo dvi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, depth, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_D32_SFLOAT,
				      { 0 }, { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 } };
	VkImageView dview;
	CHECK(vkCreateImageView(dev, &dvi, NULL, &dview));

	VkAttachmentDescription att[2] = {
		{ 0, sf.format, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		  VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR },
		{ 0, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
		  VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_UNDEFINED,
		  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL },
	};
	VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL }, dref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
	VkSubpassDescription sub = { 0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, NULL, 1, &cref, NULL, &dref, 0, NULL };
	VkSubpassDependency dep = { VK_SUBPASS_EXTERNAL, 0,
				    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
				    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
					    | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
				    0 };
	VkRenderPassCreateInfo rpi = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, NULL, 0, 2, att, 1, &sub, 1, &dep };
	VkRenderPass rp;
	CHECK(vkCreateRenderPass(dev, &rpi, NULL, &rp));
	VkFramebuffer fbs[8];
	for (uint32_t i = 0; i < nimg; i++) {
		VkImageView a2[2] = { views[i], dview };
		VkFramebufferCreateInfo fi = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, NULL, 0, rp, 2, a2, W, H, 1 };
		CHECK(vkCreateFramebuffer(dev, &fi, NULL, &fbs[i]));
	}

	VkShaderModuleCreateInfo vmi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof spv_test_vert, spv_test_vert };
	VkShaderModuleCreateInfo fmi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof spv_test_frag, spv_test_frag };
	VkShaderModule vs, fs;
	CHECK(vkCreateShaderModule(dev, &vmi, NULL, &vs));
	CHECK(vkCreateShaderModule(dev, &fmi, NULL, &fs));
	VkPushConstantRange pcr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32 };
	VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pcr };
	VkPipelineLayout pl;
	CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
	VkPipelineShaderStageCreateInfo stages[2] = {
		{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", NULL },
		{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", NULL },
	};
	VkPipelineVertexInputStateCreateInfo vin = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
	VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0,
						      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE };
	VkViewport vp = { 0, 0, W, H, 0, 1 };
	VkRect2D scissor = { { 0, 0 }, { W, H } };
	VkPipelineViewportStateCreateInfo vps = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, &vp, 1, &scissor };
	VkPipelineRasterizationStateCreateInfo ras = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
	ras.polygonMode = VK_POLYGON_MODE_FILL, ras.cullMode = VK_CULL_MODE_NONE, ras.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, ras.lineWidth = 1;
	VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
	ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineDepthStencilStateCreateInfo ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
	ds.depthTestEnable = VK_TRUE, ds.depthWriteEnable = VK_TRUE, ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
	VkPipelineColorBlendAttachmentState cba = { 0 };
	cba.colorWriteMask = 0xF;
	VkPipelineColorBlendStateCreateInfo cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
	cb.attachmentCount = 1, cb.pAttachments = &cba;
	VkGraphicsPipelineCreateInfo gpi = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
	gpi.stageCount = 2, gpi.pStages = stages, gpi.pVertexInputState = &vin, gpi.pInputAssemblyState = &ia, gpi.pViewportState = &vps;
	gpi.pRasterizationState = &ras, gpi.pMultisampleState = &ms, gpi.pDepthStencilState = &ds, gpi.pColorBlendState = &cb;
	gpi.layout = pl, gpi.renderPass = rp;
	VkPipeline pipe;
	CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipe));

	VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family };
	VkCommandPool pool;
	CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
	VkCommandBuffer cmds[RINGF];
	VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, RINGF };
	CHECK(vkAllocateCommandBuffers(dev, &cai, cmds));
	VkSemaphore acquired[RINGF], done[8];
	VkFence fences[RINGF];
	VkSemaphoreCreateInfo semi = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, VK_FENCE_CREATE_SIGNALED_BIT };
	for (int i = 0; i < RINGF; i++) {
		CHECK(vkCreateSemaphore(dev, &semi, NULL, &acquired[i]));
		CHECK(vkCreateFence(dev, &fci, NULL, &fences[i]));
	}
	for (uint32_t i = 0; i < nimg; i++) CHECK(vkCreateSemaphore(dev, &semi, NULL, &done[i]));

	LARGE_INTEGER f, t0, now;
	QueryPerformanceFrequency(&f);
	QueryPerformanceCounter(&t0);
	LONGLONG tau = (LONGLONG)(TAU_MS * 1e-3 * f.QuadPart), event_qpc = 0, dump_qpc = 0, done_qpc = 0;
	struct mouse_shm *shm = NULL;
	HANDLE dump_event = NULL;
	wchar_t name[96];
	int frame = 0, dumped = 0, asked = 0;
	double yaw_at_dump = 0.0, px_at_dump = 0.0;
	LONGLONG present_prev = 0, present_now = 0, move_from = 0;   /* move: D held since the present before */
	for (;; frame++) {
		MSG msg;
		while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) DispatchMessageW(&msg);
		int fi = frame % RINGF;
		CHECK(vkWaitForFences(dev, 1, &fences[fi], VK_TRUE, UINT64_MAX));
		CHECK(vkResetFences(dev, 1, &fences[fi]));
		uint32_t idx;
		CHECK(vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, acquired[fi], VK_NULL_HANDLE, &idx));
		QueryPerformanceCounter(&now);
		if (!shm && frame > 0) {
			_snwprintf(name, 96, L"Local\\cloud-sim-vk-mouse-%lu", GetCurrentProcessId());
			HANDLE m = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name);
			shm = m ? MapViewOfFile(m, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof *shm) : NULL;
			if (shm) shm->frequency = f.QuadPart, shm->version = MOUSE_SHM_VERSION;   /* the writer here, as input_logger */
			_snwprintf(name, 96, L"Local\\cloud-sim-vk-dump-%lu", GetCurrentProcessId());
			dump_event = OpenEventW(EVENT_MODIFY_STATE, FALSE, name);
		}
		static double seg_x[64], seg_z[64];  /* walk: key states, each held from the present before it */
		static LONGLONG seg_t[64];
		static int nseg, cur_keys = -1;
		if (walk && kbits && frame >= 30) {
			static const int plan[8] = { 8, 2, 1, 4, 9, 6, 8, 2 };   /* D, A, W, S, D+W, A+S, D, A */
			static LONGLONG w0;
			if (!w0) w0 = now.QuadPart;
			int step = (int)((now.QuadPart - w0) / (0.35 * f.QuadPart));
			int kb = step < 8 ? plan[step] : 0;
			if (kb != cur_keys && nseg < 64) {
				double fw = ((kb & 1) ? 1 : 0) - ((kb & 4) ? 1 : 0), rt = ((kb & 8) ? 1 : 0) - ((kb & 2) ? 1 : 0), nr = sqrt(fw * fw + rt * rt);
				seg_t[nseg] = present_now, seg_x[nseg] = nr > 0 ? rt / nr : 0, seg_z[nseg] = nr > 0 ? fw / nr : 0;
				nseg++;
				cur_keys = kb;
				*kbits = kb;
			}
		}
		if (move && frame == 40 && kbits && !walk) {    /* the strafe: D from now on, as the layer will see it */
			*kbits = 8;
			move_from = present_now;          /* frame 39's present: the layer holds D from there */
			event_qpc = now.QuadPart;
		}
		if (shm && pad && !padcal) {          /* 10: the stick at 0.7 right from frame 40, back 120 ms later */
			static LONGLONG held;
			if (frame == 40) pad_write(shm, now.QuadPart, 0.7, 0.0), held = event_qpc = now.QuadPart;
			if (held && now.QuadPart > held + (LONGLONG)(0.12 * f.QuadPart)) pad_write(shm, now.QuadPart, 0.0, 0.0), held = 0;
		}
		if (shm && padcal && frame >= 30) {   /* 11: the stick's path, four samples a frame */
			static LONGLONG p0, pq;
			if (!p0) p0 = pq = now.QuadPart;
			for (int j = 1; j <= 4; j++) {
				LONGLONG qj = pq + (now.QuadPart - pq) * j / 4;
				double t = (qj - p0) / (double)f.QuadPart;
				pad_write(shm, qj, pad_sx(t), pad_sy(t));
			}
			pq = now.QuadPart;
		}
		if (shm && frame == 40 && !calib && !move && !pad) {   /* the turn: one event of +COUNTS */
			struct mouse_shm_event *e = &shm->ring[shm->count & (MOUSE_SHM_EVENTS - 1)];
			e->qpc = now.QuadPart, e->dx = COUNTS, e->dy = 0;
			InterlockedIncrement64((volatile LONG64 *)&shm->count);
			event_qpc = now.QuadPart;
		}
		static LONGLONG first_turn;
		if (osc && shm && frame == 40) first_turn = now.QuadPart, event_qpc = 0;   /* the dump waits for the turn back */
		if (osc && shm && first_turn && !event_qpc && now.QuadPart > first_turn + (LONGLONG)(0.15 * f.QuadPart)) {
			struct mouse_shm_event *e = &shm->ring[shm->count & (MOUSE_SHM_EVENTS - 1)];
			e->qpc = now.QuadPart, e->dx = -COUNTS, e->dy = 0;          /* and back */
			InterlockedIncrement64((volatile LONG64 *)&shm->count);
			event_qpc = now.QuadPart;
		}
		if (shm && calib && !padcal && frame >= 30) {    /* the calibration's mouse: four events a frame */
			static LONGLONG m0, last_q;
			static long last_x, last_y;
			if (!m0) m0 = last_q = now.QuadPart, last_x = lround(cal_x(0)), last_y = lround(cal_y(0));
			for (int j = 1; j <= 4; j++) {
				LONGLONG qj = last_q + (now.QuadPart - last_q) * j / 4;
				double t = (qj - m0) / (double)f.QuadPart;
				long X = lround(cal_x(t)), Y = lround(cal_y(t));
				if (X == last_x && Y == last_y) continue;
				struct mouse_shm_event *e = &shm->ring[shm->count & (MOUSE_SHM_EVENTS - 1)];
				e->qpc = qj, e->dx = (int32_t)(X - last_x), e->dy = (int32_t)(Y - last_y);
				InterlockedIncrement64((volatile LONG64 *)&shm->count);
				last_x = X, last_y = Y;
			}
			last_q = now.QuadPart;
		}
		double counts = 0.0, county = 0.0;  /* the game's camera: the mouse up to TAU_MS ago */
		double lag = CAL_LAG_MS * 1e-3 * f.QuadPart;
		if (shm)
			for (int64_t i = 0; i < shm->count; i++) {
				struct mouse_shm_event e = shm->ring[i & (MOUSE_SHM_EVENTS - 1)];
				if (e.qpc > now.QuadPart - tau) continue;
				double w = calib ? 1.0 - exp(-(double)(now.QuadPart - tau - e.qpc) / lag) : 1.0;
				counts += e.dx * w, county += e.dy * w;
			}
		double px = 0.0, pz = 0.0;          /* move: 3 units a second, TAU_MS after the key */
		if (move && !walk && move_from && now.QuadPart - tau > move_from) {
			double T = (double)(now.QuadPart - tau - move_from) / f.QuadPart;
			px = 3.0 * (move_lag > 0 ? T - move_lag * (1.0 - exp(-T / move_lag)) : T);
		}
		if (walk)
			for (int s = 0; s < nseg; s++) {
				LONGLONG end = s + 1 < nseg ? seg_t[s + 1] : now.QuadPart - tau, lim = now.QuadPart - tau;
				if (end > lim) end = lim;
				if (end > seg_t[s]) {
					double dt = (double)(end - seg_t[s]) / f.QuadPart;
					px += 3.0 * seg_x[s] * dt, pz += 3.0 * seg_z[s] * dt;
				}
			}
		double pyaw = 0.0, ppitch = 0.0;    /* 10, 11: the stick up to TAU_MS ago */
		if (pad && shm) pad_camera(shm, now.QuadPart - tau, (double)f.QuadPart, PAD_LAG_MS * 1e-3, &pyaw, &ppitch);
		float pc[8] = { pad ? (float)pyaw : move ? (walk ? 1.0f : 0.0f) : (float)((calib ? CAL_A : A_RAD) * counts),
				pad ? (float)ppitch : (float)(calib ? CAL_B * county : 0.0),
				calib ? (float)(W * 0.5 / tan(CAL_HFOV * 3.14159265358979 / 360.0)) : W * 0.5f, (float)W, (float)H,
				calib ? 1.0f : (move ? 2.0f : 0.0f), (float)px, (float)pz };
		VkCommandBuffer cmd = cmds[fi];
		VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
		CHECK(vkBeginCommandBuffer(cmd, &bi));
		VkClearValue clear[2];
		clear[0].color = (VkClearColorValue){ { 0, 0, 0, 1 } };
		clear[1].depthStencil = (VkClearDepthStencilValue){ 1.0f, 0 };
		VkRenderPassBeginInfo rbi = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, NULL, rp, fbs[idx], { { 0, 0 }, { W, H } }, 2, clear };
		vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
		vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
		vkCmdDraw(cmd, 3, 1, 0, 0);
		vkCmdEndRenderPass(cmd);
		CHECK(vkEndCommandBuffer(cmd));
		VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 1, &acquired[fi], &wait, 1, &cmd, 1, &done[idx] };
		CHECK(vkQueueSubmit(queue, 1, &si, fences[fi]));
		/* the dump: halfway through the delay after the turn shows */
		if (event_qpc && !dumped && now.QuadPart >= event_qpc + tau + (LONGLONG)(latency * 0.5e-3 * f.QuadPart)) {
			if (dump_event) SetEvent(dump_event);
			dumped = 1, dump_qpc = now.QuadPart, yaw_at_dump = pc[0], px_at_dump = px;
		}
		VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, NULL, 1, &done[idx], 1, &swapchain, &idx, NULL };
		LARGE_INTEGER pq;
		QueryPerformanceCounter(&pq);
		present_prev = present_now, present_now = pq.QuadPart;
		CHECK(vkQueuePresentKHR(cq && frame >= 20 ? cqueue : queue, &pi));
		if (calib && frame == 40) {
			Sleep(20);
			char more[MAX_PATH * 3 + 640];
			snprintf(more, sizeof more, "%ssim_record_now=1\n", extra);
			write_settings(settings, mode, latency, more);
			asked = 1, done_qpc = now.QuadPart;
		}
		if (calib && asked && now.QuadPart > done_qpc + (LONGLONG)(6.5 * f.QuadPart)) break;
		if (walk && frame == 45) {
			Sleep(20);
			char more[MAX_PATH * 3 + 640];
			snprintf(more, sizeof more, "%ssim_demo_now=1\n", extra);
			write_settings(settings, mode, latency, more);
			asked = 1, done_qpc = now.QuadPart;
		}
		if (walk && asked && now.QuadPart > done_qpc + (LONGLONG)(5.5 * f.QuadPart)) break;
		if (!calib && !walk && dumped && !asked && frame > 0 && now.QuadPart > dump_qpc + f.QuadPart / 5) {
			Sleep(20);                    /* a new modification time */
			char more[MAX_PATH * 3 + 640];
			snprintf(more, sizeof more, "%ssim_record_now=1\nsim_snapshot_now=1\nsim_demo_now=1\nsim_demo_frames=30\n", extra);
			write_settings(settings, mode, latency, more);
			asked = 1, done_qpc = now.QuadPart;
		}
		if (!calib && !walk && asked && now.QuadPart > done_qpc + 2 * f.QuadPart) break;
		if (now.QuadPart > t0.QuadPart + 15 * f.QuadPart) {
			printf("FAIL: the run took over 15 s\n");
			break;
		}
	}
	CHECK(vkDeviceWaitIdle(dev));
	printf("%d frames, %.1f fps; turn at frame 40, dump %.0f ms after it\n", frame,
	       frame / ((now.QuadPart - t0.QuadPart) / (double)f.QuadPart), (dump_qpc - event_qpc) * 1e3 / f.QuadPart);
	Sleep(300);                                /* the layer's writer thread */

	/* tear down: the layer sees every destroy */
	for (uint32_t i = 0; i < nimg; i++) vkDestroySemaphore(dev, done[i], NULL), vkDestroyFramebuffer(dev, fbs[i], NULL), vkDestroyImageView(dev, views[i], NULL);
	for (int i = 0; i < RINGF; i++) vkDestroySemaphore(dev, acquired[i], NULL), vkDestroyFence(dev, fences[i], NULL);
	vkDestroyCommandPool(dev, pool, NULL);
	vkDestroyPipeline(dev, pipe, NULL), vkDestroyPipelineLayout(dev, pl, NULL);
	vkDestroyShaderModule(dev, vs, NULL), vkDestroyShaderModule(dev, fs, NULL);
	vkDestroyRenderPass(dev, rp, NULL);
	vkDestroyImageView(dev, dview, NULL), vkDestroyImage(dev, depth, NULL), vkFreeMemory(dev, dmem, NULL);
	vkDestroySwapchainKHR(dev, swapchain, NULL);
	vkDestroyDevice(dev, NULL);
	vkDestroySurfaceKHR(inst, surface, NULL);
	if (messenger) {
		PFN_vkDestroyDebugUtilsMessengerEXT ddm = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst, "vkDestroyDebugUtilsMessengerEXT");
		if (ddm) ddm(inst, messenger, NULL);
	}
	vkDestroyInstance(inst, NULL);
	DestroyWindow(hwnd);

	/* the checks */
	int fail = 0;
	wchar_t p[MAX_PATH];
	if (walk) {
		WIN32_FIND_DATAW wfd;
		_snwprintf(p, MAX_PATH, L"%ls\\demo-*", recdir);
		HANDLE wh = FindFirstFileW(p, &wfd);
		size_t cn = 0;
		if (wh != INVALID_HANDLE_VALUE) {
			_snwprintf(p, MAX_PATH, L"%ls\\%ls\\color.half", recdir, wfd.cFileName);
			free(slurp(p, &cn));
			FindClose(wh);
			printf("walk recording: %ls\\%ls (%d frames)\n", recdir, wfd.cFileName, (int)(cn / ((size_t)(W / 2) * (H / 2) * 4)));
			printf("truth: speed 30 near-plane units a second (3 units, near plane 0.1), tau %.1f ms; settings %ls\n", TAU_MS, settings);
		}
		int ok = cn == (size_t)300 * (W / 2) * (H / 2) * 4;
		printf(ok ? "PASSED (now run fit_move.py on it)\n" : "FAILED: no walk recording\n");
		return !ok;
	}
	if (calib) {
		WIN32_FIND_DATAW cfd;
		_snwprintf(p, MAX_PATH, L"%ls\\rec-*", recdir);
		HANDLE ch = FindFirstFileW(p, &cfd);
		size_t fr = 0;
		if (ch != INVALID_HANDLE_VALUE) {
			_snwprintf(p, MAX_PATH, L"%ls\\%ls\\frames.y8", recdir, cfd.cFileName);
			free(slurp(p, &fr));
			FindClose(ch);
			printf("calibration recording: %ls\\%ls (frames.y8 %zu bytes, %d frames)\n", recdir, cfd.cFileName, fr,
			       (int)(fr / ((size_t)(W / 4) * (H / 4))));
			if (padcal) {
				printf("truth: hfov %.1f, the stick: radial, tau %.1f ms, lag %.1f ms; in the layer's signs\n  sim_pad_yaw=", CAL_HFOV,
				       TAU_MS, PAD_LAG_MS);
				for (int i = 0; i < 11; i++) printf("%s%.4f", i ? " " : "", -PAD_YAW[i]);
				printf("\n  sim_pad_pitch=");
				for (int i = 0; i < 11; i++) printf("%s%.4f", i ? " " : "", PAD_PITCH[i]);
				printf("\n");
			} else {
				printf("truth: hfov %.1f, yaw %.6f rad/count, pitch %.6f rad/count, tau %.1f ms, lag %.1f ms\n", CAL_HFOV, CAL_A,
				       CAL_B, TAU_MS, CAL_LAG_MS);
			}
		}
		int ok = fr == (size_t)CAL_FRAMES * (W / 4) * (H / 4);
		printf(ok ? (padcal ? "PASSED (now run fit_pad.py on it)\n" : "PASSED (now run fit_doom.py on it)\n")
			  : "FAILED: no calibration recording\n");
		return !ok;
	}
	_snwprintf(p, MAX_PATH, L"%ls\\dump.txt", dump);
	char *info = (char *)slurp(p, NULL);
	_snwprintf(p, MAX_PATH, L"%ls\\out.raw", dump);
	size_t n = 0;
	unsigned char *out = slurp(p, &n);
	if (!info || !out || n != (size_t)W * H * 4) {
		printf("FAIL: no dump (%ls: %s, out.raw %zu bytes)\n", dump, info ? "info" : "no info", n);
		fail = 1;
	} else {
		printf("--- dump.txt\n%s---\n", info);
		const char *want_mode = mode == 0 ? "local" : mode == 1 ? "cloud" : "warp";
		char *m = strstr(info, "mode ");
		if (!m || strncmp(m + 5, want_mode, strlen(want_mode))) {
			printf("FAIL: the layer was not in %s\n", want_mode);
			fail = 1;
		}
		double age = 0;
		char *a = strstr(info, "src_age_ms ");
		if (a) age = atof(a + 11);
		if (mode && age < latency && !netw) {      /* with jitter a frame may well come sooner */
			printf("FAIL: the frame shown is %.1f ms old, under %d\n", age, latency);
			fail = 1;
		}
		if (cq) {
			/* on a compute-only family the layer leaves the depth out by design (no depth copies there) */
			printf("compute-only family: depth %s (none expected); the weapon, snapshot and demo checks need it and are left out\n",
			       strstr(info, "depth_ok 1") ? "present" : "absent");
			if (strstr(info, "depth_ok 1")) printf("FAIL: depth on a compute-only family\n"), fail = 1;
		} else if (!strstr(info, "depth_ok 1")) {
			printf("FAIL: the layer had no depth\n");
			fail = 1;
		}
		/* the stripe in row 40 */
		double sx = 0;
		int nr = 0;
		for (int x = 0; x < W; x++) {
			const unsigned char *q = px(out, x, 40);
			if (q[2] > 200 && q[1] < 80 && q[0] < 80) sx += x + 0.5, nr++;
		}
		/* move: the stripe is the cylinder's (0, y, 4); from (px, 0, 0) it is at x = cx - f px / 4 */
		double now_x = move ? W * 0.5 - W * 0.5 * px_at_dump / 4.0 : W * 0.5 - W * 0.5 * tan(yaw_at_dump), then_x = W * 0.5;
		if (move) {
			char *tl = strstr(info, "\nT ");
			char *sp = strstr(info, "splat ");
			printf("move: camera at x %.4f units now; the layer's move T %s; splat %c\n", px_at_dump,
			       tl ? strtok(tl + 3, "\n") : "?", sp ? sp[6] : '?');
			if (!sp || sp[6] != '1') printf("FAIL: the layer did not take the splat path\n"), fail = 1;
		}
		double want_x = mode == 1 ? then_x : now_x;
		double got_x = nr ? sx / nr : -1;
		printf("stripe at x %.2f (%d px), expected %.2f (camera now %.2f, then %.2f; yaw now %.4f)\n", got_x, nr, want_x, now_x, then_x,
		       yaw_at_dump);
		if (!nr || fabs(got_x - want_x) > 1.5) {
			printf("FAIL: the stripe is not where it should be\n");
			fail = 1;
		}
		if (osc) {   /* the left edge the old frame had turned away from: the checker's two colours, or a blur */
			int exact = 0, all = 0;
			for (int y = 30; y < 50; y++)
				for (int x = 4; x < 56; x++) {
					const unsigned char *q = px(out, x, y);
					all++;
					exact += near_rgb(q, 26, 77, 153, 6) || near_rgb(q, 153, 77, 51, 6);
				}
			printf("left edge (4..56, 30..50): %.0f %% of the pixels are the checker's own colours (%s)\n", 100.0 * exact / all,
			       nohist ? "no history: the blur" : "from an older frame");
			/* the checker's cell borders mix the two colours under bilinear sampling: 75 %, against ~2 % blurred */
			if (!nohist && exact < 0.75 * all) printf("FAIL: the edge was not filled from history\n"), fail = 1;
		}
		const unsigned char *wpx = px(out, 512, 330);
		printf("weapon (512,330): %u %u %u\n", wpx[2], wpx[1], wpx[0]);
		if (!cq && !near_rgb(wpx, 128, 128, 128, 3)) {
			printf("FAIL: the weapon moved\n");
			fail = 1;
		}
		if (mode == 2 && !cq) {
			const unsigned char *h = px(out, 420, 330);
			printf("behind the weapon (420,330): %u %u %u\n", h[2], h[1], h[0]);
			if (near_rgb(h, 128, 128, 128, 20)) {
				printf("FAIL: grey where the weapon hid the world: the weapon was warped as world\n");
				fail = 1;
			}
		}
		const unsigned char *c = px(out, 320, 200);
		printf("crosshair (320,200): %u %u %u\n", c[2], c[1], c[0]);
		if (!near_rgb(c, 255, 255, 255, 2)) {
			printf("FAIL: no crosshair\n");
			fail = 1;
		}
		const unsigned char *k = px(out, 20, 20);
		printf("mark (20,20): %u %u %u\n", k[2], k[1], k[0]);
		if ((mode == 1 && !near_rgb(k, 230, 30, 30, 2)) || (mode == 2 && !near_rgb(k, 30, 220, 60, 2))
		    || (mode == 0 && (near_rgb(k, 230, 30, 30, 2) || near_rgb(k, 30, 220, 60, 2)))) {
			printf("FAIL: the mark is wrong\n");
			fail = 1;
		}
	}
	/* the recording and the snapshot asked for through the settings */
	WIN32_FIND_DATAW fd;
	_snwprintf(p, MAX_PATH, L"%ls\\rec-*", recdir);
	HANDLE fh = FindFirstFileW(p, &fd);
	int rec_ok = 0, snap_ok = 0;
	if (fh != INVALID_HANDLE_VALUE) {
		_snwprintf(p, MAX_PATH, L"%ls\\%ls\\frames.y8", recdir, fd.cFileName);
		size_t fr = 0;
		unsigned char *y8 = slurp(p, &fr);
		_snwprintf(p, MAX_PATH, L"%ls\\%ls\\input.txt", recdir, fd.cFileName);
		char *in = (char *)slurp(p, NULL);
		const char *turn = pad ? "\np " : "\nm ";   /* the stick's samples, or the mouse's event */
		rec_ok = y8 && fr == (size_t)30 * (W / 4) * (H / 4) && in && strstr(in, turn);
		printf("recording %ls: frames.y8 %zu bytes (want %d), input.txt %s\n", fd.cFileName, fr, 30 * (W / 4) * (H / 4),
		       in && strstr(in, turn) ? "has the turn" : "without the turn");
		free(y8), free(in);
		FindClose(fh);
	}
	_snwprintf(p, MAX_PATH, L"%ls\\snap-*", recdir);
	fh = FindFirstFileW(p, &fd);
	if (fh != INVALID_HANDLE_VALUE) {
		size_t cn = 0, dn = 0;
		_snwprintf(p, MAX_PATH, L"%ls\\%ls\\color.raw", recdir, fd.cFileName);
		unsigned char *col = slurp(p, &cn);
		_snwprintf(p, MAX_PATH, L"%ls\\%ls\\depth.raw", recdir, fd.cFileName);
		unsigned char *dep = slurp(p, &dn);
		float dv = 0;
		if (dep && dn == (size_t)W * H * 4) memcpy(&dv, dep + ((size_t)330 * W + 512) * 4, 4);
		float dw = 0;
		if (dep && dn == (size_t)W * H * 4) memcpy(&dw, dep + ((size_t)40 * W + 100) * 4, 4);
		snap_ok = col && cn == (size_t)W * H * 4 && dep && dn == (size_t)W * H * 4 && fabsf(dv - 0.05f) < 1e-4f && fabsf(dw - 0.9f) < 1e-4f;
		printf("snapshot %ls: color %zu bytes, depth %zu bytes, depth at the weapon %.4f, at the world %.4f\n", fd.cFileName, cn, dn, dv, dw);
		free(col), free(dep);
		FindClose(fh);
	}
	/* the demo recording: 30 frames of half-size colour and nearness */
	int demo_ok = 0;
	_snwprintf(p, MAX_PATH, L"%ls\\demo-*", recdir);
	fh = FindFirstFileW(p, &fd);
	if (fh != INVALID_HANDLE_VALUE) {
		size_t cn = 0, nn = 0;
		_snwprintf(p, MAX_PATH, L"%ls\\%ls\\color.half", recdir, fd.cFileName);
		unsigned char *col = slurp(p, &cn);
		_snwprintf(p, MAX_PATH, L"%ls\\%ls\\near.half", recdir, fd.cFileName);
		unsigned char *nr = slurp(p, &nn);
		_snwprintf(p, MAX_PATH, L"%ls\\%ls\\times.txt", recdir, fd.cFileName);
		char *tm = (char *)slurp(p, NULL);
		size_t hw = W / 2, hh = H / 2;
		int frames_c = (int)(cn / (hw * hh * 4)), frames_n = (int)(nn / (hw * hh * 2));
		unsigned near_weapon = 0, near_world = 0;
		const unsigned char *gpx = NULL;
		if (nr && frames_n > 0) {
			const unsigned short *u = (const unsigned short *)nr;
			near_weapon = u[(size_t)165 * hw + 256];          /* (512, 330) at full size */
			near_world = u[(size_t)20 * hw + 50];             /* (100, 40) */
		}
		if (col && frames_c > 0) gpx = col + ((size_t)165 * hw + 256) * 4;
		int lines = 0;
		for (char *c = tm; c && *c; c++) lines += *c == '\n';
		demo_ok = frames_c == 30 && frames_n == 30 && lines == 31 && abs((int)near_weapon - 3277) < 40
			  && abs((int)near_world - 58982) < 40 && gpx && near_rgb(gpx, 128, 128, 128, 3);
		printf("demo %ls: color.half %d frames, near.half %d frames, times %d lines; nearness at the weapon %u (0.05 -> 3277), "
		       "at the world %u (0.9 -> 58982); colour at the weapon %u %u %u\n",
		       fd.cFileName, frames_c, frames_n, lines - 1, near_weapon, near_world, gpx ? gpx[2] : 0, gpx ? gpx[1] : 0,
		       gpx ? gpx[0] : 0);
		free(col), free(nr), free(tm);
		FindClose(fh);
	}
	if (move) {
		printf("(the recording, snapshot and demo checks expect the panorama's depth 0.9: left to the other modes)\n");
	} else if (cq) {
		if (!rec_ok) printf("FAIL: the recording\n"), fail = 1;
	} else {
		if (!rec_ok) printf("FAIL: the recording\n"), fail = 1;
		if (!snap_ok) printf("FAIL: the snapshot\n"), fail = 1;
		if (!demo_ok) printf("FAIL: the demo recording\n"), fail = 1;
	}
	if (validation) {
		printf("validation: %d errors, %d warnings\n", errors, warnings);
		if (errors) fail = 1;
	}
	char *lg = (char *)slurp(logpath, NULL);
	printf("--- layer log\n%s---\n", lg ? lg : "(none)");
	printf(fail ? "FAILED\n" : "PASSED\n");
	return fail;
}
