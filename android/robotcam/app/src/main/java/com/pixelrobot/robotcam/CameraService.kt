package com.pixelrobot.robotcam

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.SystemClock
import android.system.ErrnoException
import android.system.Os
import android.util.Log
import android.util.Range
import android.util.Size
import java.io.File
import java.io.IOException
import java.security.SecureRandom
import java.util.concurrent.Executor

/**
 * Keeps the back camera open and about once a second publishes the newest JPEG to
 * Download/robotcam/frame.jpg plus a frame.json sidecar, each via temp file + rename.
 *
 * Mode A: repeating JPEG stream; the writer publishes the newest frame once a second.
 * Mode B (default): repeating preview to a small YUV surface keeps exposure and focus
 * converged; one still JPEG capture per second is published as soon as it arrives.
 *
 * Published files are deleted before every (re)start, on any camera error, and when the
 * service stops; nothing is published once stopping has begun.
 */
class CameraService : Service() {

    private class Frame(val jpeg: ByteArray, val captureWallMs: Long, val seq: Long)

    // Guards latest, publishing and the published files.
    private val lock = Object()
    private var latest: Frame? = null
    private var publishing = false

    @Volatile private var running = false
    @Volatile private var mode = MODE_B
    private var sessionId = ""

    // Camera state; touched only on camThread.
    private lateinit var camThread: HandlerThread
    private lateinit var camHandler: Handler
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var jpegReader: ImageReader? = null
    private var previewReader: ImageReader? = null
    private var cameraSeq = 0L
    private var stillInFlight = false
    private val reopen = Runnable { openCamera() }
    private val stillTick = Runnable { captureStill() }

    // Chosen configuration, reported in the log, the notification and the sidecar.
    @Volatile private var size = Size(0, 0)
    @Volatile private var sizeRule = ""
    @Volatile private var jpegSizes = emptyList<Size>()
    @Volatile private var fps = Range(0, 0)
    @Volatile private var realtimeTimestamps = false

    private var writer: Thread? = null
    @Volatile private var framesWritten = 0L
    @Volatile private var writeErrors = 0L
    @Volatile private var lastError = ""

    private val outDir = File(
        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "robotcam"
    )

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Every startForegroundService() call must be answered with startForeground().
        try {
            if (Build.VERSION.SDK_INT >= 30) {
                startForeground(NOTIFICATION_ID, notification("starting"),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA)
            } else {
                startForeground(NOTIFICATION_ID, notification("starting"))
            }
        } catch (e: SecurityException) {
            Log.i(TAG, "ERROR startForeground refused", e)
            stopSelf()
            return START_NOT_STICKY
        } catch (e: IllegalStateException) {
            Log.i(TAG, "ERROR startForeground refused", e)
            stopSelf()
            return START_NOT_STICKY
        }
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "ERROR CAMERA permission not granted; open the RobotCam app once to grant it")
            stopSelf()
            return START_NOT_STICKY
        }
        val requested = if (intent?.getStringExtra(EXTRA_MODE).equals(MODE_A, ignoreCase = true)) {
            MODE_A
        } else MODE_B
        if (!running) {
            running = true
            mode = requested
            sessionId = newSessionId()
            synchronized(lock) {
                latest = null
                publishing = true
                deletePublished()
            }
            Log.i(TAG, "start: session $sessionId, mode $mode")
            camThread = HandlerThread("robotcam-camera").also { it.start() }
            camHandler = Handler(camThread.looper)
            camHandler.post { openCamera() }
            writer = Thread(::writeLoop, "robotcam-writer").also { it.start() }
        } else if (requested != mode) {
            camHandler.post {
                Log.i(TAG, "switching mode $mode -> $requested")
                mode = requested
                restartCamera("mode switch")
            }
        }
        // Not sticky: a restart by the system would come from the background, where the
        // camera is not available to a foreground service anyway.
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        if (running) {
            running = false
            // After this block the writer can no longer publish, and the files are gone.
            synchronized(lock) {
                publishing = false
                latest = null
                deletePublished()
                lock.notifyAll()
            }
            writer?.interrupt()
            camHandler.post { closeCamera() }
            camThread.quitSafely()
            Log.i(TAG, "stopped: session $sessionId, frames written $framesWritten")
        }
        super.onDestroy()
    }

    // ---------------------------------------------------------------- camera (camThread)

    private fun openCamera() {
        if (!running || device != null) return
        val manager = getSystemService(CameraManager::class.java)
        try {
            val id = manager.cameraIdList.first {
                manager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
            }
            val chars = manager.getCameraCharacteristics(id)
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
            jpegSizes = map.getOutputSizes(ImageFormat.JPEG).toList()
            val (chosen, rule) = chooseJpegSize(jpegSizes)
            size = chosen
            sizeRule = rule
            fps = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)!!
                .minWith(compareBy<Range<Int>>({ it.upper }, { it.lower }))
            realtimeTimestamps = chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
            val orientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
            Log.i(TAG, "camera $id JPEG output sizes: ${jpegSizes.joinToString(" ")}")
            Log.i(TAG, "chosen jpeg ${size.width}x${size.height} ($rule), mode $mode, fps $fps, " +
                "sensor orientation $orientation, realtime timestamps $realtimeTimestamps")

            jpegReader = ImageReader.newInstance(size.width, size.height, ImageFormat.JPEG, 2).apply {
                setOnImageAvailableListener({ r -> onJpeg(r) }, camHandler)
            }
            if (mode == MODE_B) {
                val yuv = chooseYuvSize(map.getOutputSizes(ImageFormat.YUV_420_888).toList(), size)
                Log.i(TAG, "preview surface YUV ${yuv.width}x${yuv.height}")
                previewReader = ImageReader.newInstance(yuv.width, yuv.height,
                    ImageFormat.YUV_420_888, 2).apply {
                    // Only there so 3A has a running stream; drop every preview frame.
                    setOnImageAvailableListener({ r -> r.acquireLatestImage()?.close() }, camHandler)
                }
            }
            manager.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    if (!running) { camera.close(); return }
                    device = camera
                    try {
                        startSession(camera, orientation)
                    } catch (e: Exception) {
                        failed("createCaptureSession", e)
                    }
                }
                override fun onDisconnected(camera: CameraDevice) = lost(camera, "disconnected")
                override fun onError(camera: CameraDevice, error: Int) = lost(camera, "error $error")
            }, camHandler)
        } catch (e: Exception) {
            // CameraAccessException, SecurityException, IllegalArgumentException, no back camera.
            failed("openCamera", e)
        }
    }

    private fun startSession(camera: CameraDevice, orientation: Int) {
        val jpegSurface = jpegReader!!.surface
        val previewSurface = previewReader?.surface
        val callback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) {
                if (!running) { s.close(); return }
                session = s
                try {
                    if (mode == MODE_A) {
                        s.setRepeatingRequest(request(camera, CameraDevice.TEMPLATE_PREVIEW,
                            jpegSurface, orientation), null, camHandler)
                    } else {
                        val preview = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(previewSurface!!)
                            set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
                        }.build()
                        s.setRepeatingRequest(preview, null, camHandler)
                        stillRequest = request(camera, CameraDevice.TEMPLATE_STILL_CAPTURE,
                            jpegSurface, orientation)
                        stillInFlight = false
                        camHandler.removeCallbacks(stillTick)
                        camHandler.postDelayed(stillTick, WRITE_PERIOD_MS)
                    }
                    status("streaming")
                } catch (e: Exception) {
                    failed("start requests", e)
                }
            }
            override fun onConfigureFailed(s: CameraCaptureSession) {
                failed("session configuration", null)
            }
        }
        val outputs = listOfNotNull(jpegSurface, previewSurface).map { OutputConfiguration(it) }
        camera.createCaptureSession(SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR, outputs, Executor { camHandler.post(it) }, callback
        ))
    }

    private var stillRequest: CaptureRequest? = null

    private fun request(camera: CameraDevice, template: Int, target: android.view.Surface,
                        orientation: Int): CaptureRequest =
        camera.createCaptureRequest(template).apply {
            addTarget(target)
            set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
            set(CaptureRequest.JPEG_QUALITY, JPEG_QUALITY)
            // Same as termux-camera-photo with the phone upright.
            set(CaptureRequest.JPEG_ORIENTATION, orientation)
        }.build()

    /** Mode B: one still JPEG per period; skipped while the previous one is still in flight. */
    private fun captureStill() {
        val s = session ?: return
        val req = stillRequest ?: return
        camHandler.postDelayed(stillTick, WRITE_PERIOD_MS)
        if (stillInFlight) return
        try {
            stillInFlight = true
            s.capture(req, object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult
                ) { stillInFlight = false }
                override fun onCaptureFailed(
                    session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure
                ) {
                    stillInFlight = false
                    Log.i(TAG, "ERROR still capture failed, reason ${failure.reason}")
                }
            }, camHandler)
        } catch (e: Exception) {
            failed("still capture", e)
        }
    }

    private fun onJpeg(r: ImageReader) {
        val image = r.acquireLatestImage() ?: return
        image.use {
            val buffer = it.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            val now = System.currentTimeMillis()
            // Convert the sensor timestamp to wall clock when it shares the elapsedRealtime base;
            // otherwise fall back to arrival time (later than capture by the pipeline latency).
            val wall = if (realtimeTimestamps) {
                now - (SystemClock.elapsedRealtimeNanos() - it.timestamp) / 1_000_000
            } else now
            synchronized(lock) {
                if (!publishing) return
                latest = Frame(bytes, wall, ++cameraSeq)
                lock.notifyAll()
            }
        }
    }

    private fun lost(camera: CameraDevice, why: String) {
        camera.close()
        if (device === camera) device = null
        failed("camera $why", null)
    }

    /** Any camera failure: unpublish, release everything, retry after a delay. */
    private fun failed(what: String, e: Exception?) {
        Log.i(TAG, "ERROR $what failed${e?.let { ": $it" } ?: ""}", e)
        status("$what failed, retrying")
        unpublish()
        closeCamera()
        scheduleReopen()
    }

    private fun restartCamera(why: String) {
        Log.i(TAG, "restarting camera: $why")
        unpublish()
        closeCamera()
        camHandler.removeCallbacks(reopen)
        openCamera()
    }

    private fun scheduleReopen() {
        camHandler.removeCallbacks(reopen)
        if (running) camHandler.postDelayed(reopen, REOPEN_DELAY_MS)
    }

    private fun closeCamera() {
        camHandler.removeCallbacks(stillTick)
        stillRequest = null
        stillInFlight = false
        try { session?.close() } catch (e: IllegalStateException) { /* already closed */ }
        session = null
        device?.close(); device = null
        jpegReader?.close(); jpegReader = null
        previewReader?.close(); previewReader = null
    }

    // ---------------------------------------------------------------- publishing

    /** Removes the published files and the pending frame, so no reader sees an old frame. */
    private fun unpublish() {
        synchronized(lock) {
            latest = null
            deletePublished()
        }
    }

    /** Caller holds [lock]. */
    private fun deletePublished() {
        File(outDir, SIDECAR_NAME).delete()
        File(outDir, JPEG_NAME).delete()
    }

    private fun writeLoop() {
        var lastSeq = 0L
        var nextDue = 0L
        while (true) {
            // Wait for a frame newer than the last one published; in mode A also for the period.
            val f = synchronized(lock) {
                var pick: Frame? = null
                while (pick == null) {
                    if (!publishing) return
                    val cur = latest
                    val wait = when {
                        cur == null || cur.seq == lastSeq -> 0L
                        mode == MODE_A && SystemClock.elapsedRealtime() < nextDue ->
                            nextDue - SystemClock.elapsedRealtime()
                        else -> { pick = cur; break }
                    }
                    try { lock.wait(wait) } catch (e: InterruptedException) { return }
                }
                pick!!
            }
            lastSeq = f.seq
            nextDue = SystemClock.elapsedRealtime() + WRITE_PERIOD_MS
            try {
                publish(f)
            } catch (e: IOException) {
                writeFailed(e)
            } catch (e: ErrnoException) {
                writeFailed(e)
            }
        }
    }

    private fun publish(f: Frame) {
        val n = framesWritten + 1
        val jpeg = withComment(f.jpeg,
            "robotcam session=$sessionId frame=$n capture_wall_ms=${f.captureWallMs}")
        val json = """{"session_id":"$sessionId","frame":$n,"capture_wall_ms":${f.captureWallMs},""" +
            """"written_wall_ms":${System.currentTimeMillis()},"mode":"$mode",""" +
            """"width":${size.width},"height":${size.height},"size_rule":"$sizeRule",""" +
            """"bytes":${jpeg.size},"fps_min":${fps.lower},"fps_max":${fps.upper},""" +
            """"timestamp_source":"${if (realtimeTimestamps) "sensor" else "arrival"}",""" +
            """"jpeg_sizes":[${jpegSizes.joinToString(",") { "\"${it.width}x${it.height}\"" }}]}""" +
            "\n"
        synchronized(lock) {
            // Stop, a camera error or a restart since this frame was picked: do not publish it.
            if (!publishing || latest !== f) return
            outDir.mkdirs()
            // Image first, sidecar last: a sidecar always describes an image that is in place.
            writeAtomic(JPEG_NAME, jpeg)
            writeAtomic(SIDECAR_NAME, json.toByteArray())
        }
        framesWritten = n
        if (n == 1L || n % 10 == 0L) status("streaming")
    }

    private fun writeAtomic(name: String, bytes: ByteArray) {
        val tmp = File(outDir, "$name.tmp")
        tmp.writeBytes(bytes)
        Os.rename(tmp.path, File(outDir, name).path)
    }

    private fun writeFailed(e: Exception) {
        writeErrors++
        lastError = e.message ?: e.javaClass.simpleName
        Log.i(TAG, "ERROR write failed: $lastError")
        if (writeErrors == 1L || writeErrors % 10 == 0L) status("write failed")
    }

    // ---------------------------------------------------------------- notification

    private fun status(state: String) {
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(state))
    }

    private fun notification(state: String): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Robot camera", NotificationManager.IMPORTANCE_LOW)
        )
        val detail = buildString {
            append("mode $mode, ${size.width}x${size.height}, fps ${fps.lower}-${fps.upper}, ")
            append("written $framesWritten")
            if (writeErrors > 0) append(", write errors $writeErrors: $lastError")
        }
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentTitle("RobotCam: $state")
            .setContentText(detail)
            .setStyle(Notification.BigTextStyle().bigText(detail))
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val TAG = "RobotCam"
        private const val CHANNEL_ID = "robotcam"
        private const val NOTIFICATION_ID = 1
        private const val JPEG_QUALITY: Byte = 85
        private const val WRITE_PERIOD_MS = 1000L
        private const val REOPEN_DELAY_MS = 2000L
        private const val JPEG_NAME = "frame.jpg"
        private const val SIDECAR_NAME = "frame.json"
        const val EXTRA_MODE = "mode"
        const val MODE_A = "A"
        const val MODE_B = "B"

        private fun newSessionId(): String {
            val b = ByteArray(8)
            SecureRandom().nextBytes(b)
            return b.joinToString("") { "%02x".format(it) }
        }

        /**
         * Exactly 640x480 if offered; otherwise the smallest 4:3 size of at least 640x480;
         * otherwise the smallest size at least 640 wide; otherwise the largest size.
         */
        fun chooseJpegSize(sizes: List<Size>): Pair<Size, String> {
            sizes.firstOrNull { it.width == 640 && it.height == 480 }?.let { return it to "exact 640x480" }
            val byArea = sizes.sortedBy { it.width.toLong() * it.height }
            byArea.firstOrNull { it.width * 3 == it.height * 4 && it.width >= 640 && it.height >= 480 }
                ?.let { return it to "smallest 4:3 >= 640x480" }
            byArea.firstOrNull { it.width >= 640 }?.let { return it to "smallest >= 640 wide" }
            return byArea.last() to "largest (none >= 640 wide)"
        }

        /** Smallest preview size with the JPEG's aspect ratio (same field of view), else smallest. */
        fun chooseYuvSize(sizes: List<Size>, jpeg: Size): Size {
            val byArea = sizes.sortedBy { it.width.toLong() * it.height }
            return byArea.firstOrNull {
                it.width.toLong() * jpeg.height == it.height.toLong() * jpeg.width && it.width >= 320
            } ?: byArea.first()
        }

        /**
         * Inserts a JPEG COM segment after the APPn segments, so the reader can check that the
         * image and the sidecar belong to the same frame. Returns [jpeg] unchanged if it does not
         * look like a JPEG.
         */
        fun withComment(jpeg: ByteArray, text: String): ByteArray {
            if (jpeg.size < 4 || jpeg[0] != 0xFF.toByte() || jpeg[1] != 0xD8.toByte()) return jpeg
            var pos = 2
            while (pos + 4 <= jpeg.size && jpeg[pos] == 0xFF.toByte() &&
                (jpeg[pos + 1].toInt() and 0xF0) == 0xE0
            ) {
                val len = ((jpeg[pos + 2].toInt() and 0xFF) shl 8) or (jpeg[pos + 3].toInt() and 0xFF)
                pos += 2 + len
            }
            if (pos > jpeg.size) return jpeg
            val body = text.toByteArray(Charsets.US_ASCII)
            val segLen = body.size + 2
            val segment = byteArrayOf(0xFF.toByte(), 0xFE.toByte(),
                (segLen shr 8).toByte(), (segLen and 0xFF).toByte()) + body
            return jpeg.copyOfRange(0, pos) + segment + jpeg.copyOfRange(pos, jpeg.size)
        }
    }
}
