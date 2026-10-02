package com.pixelrobot.robotcam

import android.content.Intent
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject

/** Test-only options. The default profile does not set any additional request key. */
data class CameraPower(
    val template: String = "still", val processing: String = "default",
    val focus: Float? = null, val frameMs: Int = 0, val cameraId: String = "",
    val dump: Boolean = false
) {
    private var chars: CameraCharacteristics? = null
    private var manualExposure: Long? = null
    private var manualIso: Int? = null
    private var phaseAt = 0L
    private var active = false
    private val overrides = linkedMapOf<CaptureRequest.Key<Int>, Int>()
    @Volatile private var observation: Pair<CaptureRequest, TotalCaptureResult>? = null
    private var selectedId = ""

    val variant get() = "template=$template;processing=$processing;focus=${focus ?: "auto"};frame_ms=$frameMs;camera=${cameraId.ifEmpty { "default" }}"
    val stillTemplate get() = when (template) {
        "preview" -> CameraDevice.TEMPLATE_PREVIEW
        "record" -> CameraDevice.TEMPLATE_RECORD
        else -> CameraDevice.TEMPLATE_STILL_CAPTURE
    }

    fun configure(c: CameraCharacteristics, id: String) {
        chars = c
        selectedId = id
        active = false
        manualExposure = null
        manualIso = null
        phaseAt = SystemClock.elapsedRealtime()
        observation = null
        overrides.clear()
        if (focus != null) {
            require(c.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)?.contains(CaptureRequest.CONTROL_AF_MODE_OFF) == true) { "AF OFF unavailable" }
            require(focus <= (c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f)) { "focus_diopters exceeds camera limit" }
        }
        if (frameMs != 0) {
            require(c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true) { "manual sensor unsupported" }
            require(c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES)?.contains(CaptureRequest.CONTROL_AE_MODE_OFF) == true) { "AE OFF unavailable" }
            require(frameMs * 1_000_000L <= (c.get(CameraCharacteristics.SENSOR_INFO_MAX_FRAME_DURATION) ?: 0L)) { "frame_ms exceeds sensor maximum" }
        }
        if (processing != "default") {
            val wanted = if (processing == "off") 0 else 1 // OFF/FAST for these advertised mode families
            fun choose(key: CaptureRequest.Key<Int>, modes: IntArray?, preferred: Int) {
                if (modes?.contains(preferred) == true) overrides[key] = preferred
                else if (modes?.contains(1) == true && preferred == 0) overrides[key] = 1
            }
            choose(CaptureRequest.NOISE_REDUCTION_MODE, c.get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES), wanted)
            choose(CaptureRequest.EDGE_MODE, c.get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES), wanted)
            choose(CaptureRequest.HOT_PIXEL_MODE, c.get(CameraCharacteristics.HOT_PIXEL_AVAILABLE_HOT_PIXEL_MODES), wanted)
            choose(CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE, c.get(CameraCharacteristics.COLOR_CORRECTION_AVAILABLE_ABERRATION_MODES), wanted)
            choose(CaptureRequest.SHADING_MODE, c.get(CameraCharacteristics.SHADING_AVAILABLE_MODES), wanted)
            choose(CaptureRequest.TONEMAP_MODE, c.get(CameraCharacteristics.TONEMAP_AVAILABLE_TONE_MAP_MODES), CaptureRequest.TONEMAP_MODE_FAST)
            choose(CaptureRequest.STATISTICS_FACE_DETECT_MODE, c.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_FACE_DETECT_MODES), CaptureRequest.STATISTICS_FACE_DETECT_MODE_OFF)
            choose(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION), CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)
            choose(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, c.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES), CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
        }
    }

    fun apply(b: CaptureRequest.Builder) {
        overrides.forEach { (k, v) -> b.set(k, v) }
        focus?.let {
            b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            b.set(CaptureRequest.LENS_FOCUS_DISTANCE, it)
        }
        if (frameMs != 0) {
            b.set(CaptureRequest.CONTROL_AE_MODE, if (active) CaptureRequest.CONTROL_AE_MODE_OFF else CaptureRequest.CONTROL_AE_MODE_ON)
            if (active) {
                b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, manualExposure!!)
                b.set(CaptureRequest.SENSOR_SENSITIVITY, manualIso!!)
                b.set(CaptureRequest.SENSOR_FRAME_DURATION, maxOf(frameMs * 1_000_000L, manualExposure!!))
            }
        }
    }

    /** Returns true only when the opt-in manual phase needs new requests. Called on camera thread. */
    fun observe(request: CaptureRequest, result: TotalCaptureResult): Boolean {
        observation = request to result
        if (frameMs == 0) return false
        val now = SystemClock.elapsedRealtime()
        // Ignore results queued from the previous phase when replacing the repeating request.
        val resultAe = result.get(CaptureResult.CONTROL_AE_MODE)
        if (active && now - phaseAt >= 30_000 && resultAe == CaptureRequest.CONTROL_AE_MODE_OFF) {
            active = false
            phaseAt = now
            return true
        }
        val state = result.get(CaptureResult.CONTROL_AE_STATE)
        if (!active && now - phaseAt >= 1_000 && resultAe == CaptureRequest.CONTROL_AE_MODE_ON &&
            (state == CaptureResult.CONTROL_AE_STATE_CONVERGED || state == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED)) {
            val exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: return false
            val iso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: return false
            val c = chars!!
            if (c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)?.contains(exposure) != true ||
                c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)?.contains(iso) != true) return false
            // Keep converged exposure; a longer frame period must not introduce extra motion blur.
            manualExposure = exposure
            manualIso = iso
            active = true
            phaseAt = now
            return true
        }
        return false
    }

    // Serialize only on publication, not on every dropped preview frame.
    private fun observationJson(request: CaptureRequest, result: TotalCaptureResult): String {
        val requested = JSONObject()
        val applied = JSONObject()
        fun <T> field(name: String, rk: CaptureRequest.Key<T>, sk: CaptureResult.Key<T>) {
            requested.put(name, request.get(rk) ?: JSONObject.NULL)
            applied.put(name, result.get(sk) ?: JSONObject.NULL)
        }
        field("frame_duration_ns", CaptureRequest.SENSOR_FRAME_DURATION, CaptureResult.SENSOR_FRAME_DURATION)
        field("exposure_ns", CaptureRequest.SENSOR_EXPOSURE_TIME, CaptureResult.SENSOR_EXPOSURE_TIME)
        field("iso", CaptureRequest.SENSOR_SENSITIVITY, CaptureResult.SENSOR_SENSITIVITY)
        field("ae_mode", CaptureRequest.CONTROL_AE_MODE, CaptureResult.CONTROL_AE_MODE)
        field("af_mode", CaptureRequest.CONTROL_AF_MODE, CaptureResult.CONTROL_AF_MODE)
        field("nr_mode", CaptureRequest.NOISE_REDUCTION_MODE, CaptureResult.NOISE_REDUCTION_MODE)
        field("edge_mode", CaptureRequest.EDGE_MODE, CaptureResult.EDGE_MODE)
        field("focus_diopters", CaptureRequest.LENS_FOCUS_DISTANCE, CaptureResult.LENS_FOCUS_DISTANCE)
        field("hot_pixel_mode", CaptureRequest.HOT_PIXEL_MODE, CaptureResult.HOT_PIXEL_MODE)
        field("aberration_mode", CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE, CaptureResult.COLOR_CORRECTION_ABERRATION_MODE)
        field("shading_mode", CaptureRequest.SHADING_MODE, CaptureResult.SHADING_MODE)
        field("tonemap_mode", CaptureRequest.TONEMAP_MODE, CaptureResult.TONEMAP_MODE)
        field("face_detect_mode", CaptureRequest.STATISTICS_FACE_DETECT_MODE, CaptureResult.STATISTICS_FACE_DETECT_MODE)
        field("ois_mode", CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureResult.LENS_OPTICAL_STABILIZATION_MODE)
        field("video_stabilization_mode", CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE)
        field("capture_intent", CaptureRequest.CONTROL_CAPTURE_INTENT, CaptureResult.CONTROL_CAPTURE_INTENT)
        return JSONObject().put("requested", requested).put("result", applied)
            .put("result_sensor_timestamp_ns", result.get(CaptureResult.SENSOR_TIMESTAMP) ?: JSONObject.NULL)
            .put("result_frame_number", result.frameNumber)
            .put("manual_phase", if (frameMs == 0) "disabled" else if (result.get(CaptureResult.CONTROL_AE_MODE) == CaptureRequest.CONTROL_AE_MODE_OFF) "manual" else "ae_converging")
            .put("camera_id", selectedId).toString()
    }

    fun diagnostics(): String = (observation?.let { JSONObject(observationJson(it.first, it.second)) }
        ?: JSONObject()).put("options", JSONObject()
        .put("capture_template", template).put("processing", processing)
        .put("focus_diopters", focus ?: JSONObject.NULL).put("frame_ms", frameMs)
        .put("camera_id", cameraId).put("dump_characteristics", dump)).toString()

    companion object {
        val EXTRAS = listOf("capture_template", "processing", "focus_diopters", "frame_ms", "camera_id", "dump_characteristics")
        fun from(intent: Intent?): CameraPower {
            val p = CameraPower(intent?.getStringExtra("capture_template") ?: "still",
                intent?.getStringExtra("processing") ?: "default",
                if (intent?.hasExtra("focus_diopters") == true) intent.getFloatExtra("focus_diopters", -1f) else null,
                intent?.getIntExtra("frame_ms", 0) ?: 0, intent?.getStringExtra("camera_id") ?: "",
                intent?.getBooleanExtra("dump_characteristics", false) ?: false)
            require(p.template in listOf("still", "preview", "record")) { "invalid capture_template" }
            require(p.processing in listOf("default", "fast", "off")) { "invalid processing" }
            require(p.focus == null || (p.focus.isFinite() && p.focus >= 0f)) { "invalid focus_diopters" }
            require(p.frameMs == 0 || p.frameMs in 200..500) { "frame_ms must be 0 or 200..500" }
            return p
        }

        fun characteristics(manager: CameraManager): String {
            val rows = JSONArray()
            for (id in manager.cameraIdList) {
                val c = manager.getCameraCharacteristics(id)
                val r = JSONObject().put("camera_id", id)
                fun <T> item(name: String, key: CameraCharacteristics.Key<T>) {
                    val value = c.get(key)
                    r.put(name, when (value) {
                        is IntArray -> JSONArray(value.toList())
                        is FloatArray -> JSONArray(value.toList())
                        else -> value?.toString() ?: JSONObject.NULL
                    })
                }
                item("facing", CameraCharacteristics.LENS_FACING)
                item("focal_lengths_mm", CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                item("min_focus_diopters", CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)
                item("capabilities", CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                item("nr_modes", CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES)
                item("edge_modes", CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES)
                item("ois_modes", CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
                item("max_frame_duration_ns", CameraCharacteristics.SENSOR_INFO_MAX_FRAME_DURATION)
                r.put("fps_ranges", JSONArray(c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.map { it.toString() } ?: emptyList<String>()))
                r.put("session_keys", JSONArray(c.availableSessionKeys?.map { it.name } ?: emptyList<String>()))
                r.put("physical_ids", JSONArray(c.physicalCameraIds.toList()))
                val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                val sizes = JSONArray()
                for (s in map?.getOutputSizes(ImageFormat.YUV_420_888) ?: emptyArray()) {
                    sizes.put(JSONObject().put("width", s.width).put("height", s.height)
                        .put("min_frame_duration_ns", map!!.getOutputMinFrameDuration(ImageFormat.YUV_420_888, s)))
                }
                r.put("yuv_sizes", sizes)
                rows.put(r)
            }
            return JSONObject().put("cameras", rows).toString(2)
        }
    }
}
