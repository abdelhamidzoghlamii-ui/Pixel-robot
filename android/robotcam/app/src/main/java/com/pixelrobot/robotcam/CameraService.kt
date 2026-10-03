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
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
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
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.security.SecureRandom
import java.util.concurrent.Executor

/**
 * Keeps the back camera open and publishes a 640x480 JPEG to Download/robotcam/frame.jpg plus
 * a frame.json sidecar, each via temp file + rename, 1 or 2 times per second.
 *
 * Frames come from a YUV_420_888 stream at 640x480 (or the nearest 4:3 size) and are
 * compressed to JPEG in the app, only when they are published.
 *
 * Mode A: repeating YUV stream; a frame is copied out and published once per period.
 * Mode B (default): repeating preview to a small YUV surface keeps exposure and focus
 * converged; one still YUV capture per period is published as soon as it arrives.
 *
 * Published files are deleted before every (re)start, on any camera error, and when the
 * service stops; nothing is published once stopping has begun.
 */
class CameraService : Service() {

    private class Frame(
        val nv21: ByteArray, val width: Int, val height: Int,
        val captureBootMs: Long, val captureWallMs: Long, val seq: Long
    )

    // Guards latest, publishing and the published files.
    private val lock = Object()
    private var latest: Frame? = null
    private var publishing = false

    @Volatile private var running = false
    @Volatile private var mode = MODE_B
    @Volatile private var rate = 1
    private val periodMs get() = 1000L / rate
    private var sessionId = ""
    @Volatile private var power = CameraPower()

    // Camera state; touched only on camThread.
    private lateinit var camThread: HandlerThread
    private lateinit var camHandler: Handler
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var frameReader: ImageReader? = null
    private var previewReader: ImageReader? = null
    private var stillRequest: CaptureRequest? = null
    private var cameraSeq = 0L
    private var nextStreamDue = 0L
    private var stillInFlight = false
    private var gen = 0L          // current open attempt; see the camera section
    private var opening = false   // an openCamera() of the current attempt is pending
    private val reopen = Runnable { openCamera() }
    private val stillTick = Runnable { captureStill() }

    // Chosen configuration, reported in the log, the notification and the sidecar.
    @Volatile private var size = Size(0, 0)
    @Volatile private var sizeRule = ""
    @Volatile private var yuvSizes = emptyList<Size>()
    @Volatile private var fps = Range(0, 0)
    @Volatile private var exifOrientation = 1
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
        val reqMode = if (intent?.getStringExtra(EXTRA_MODE).equals(MODE_A, ignoreCase = true)) {
            MODE_A
        } else MODE_B
        val askedRate = intent?.getIntExtra(EXTRA_RATE, 1) ?: 1
        val reqRate = if (askedRate == 2) 2 else 1
        if (askedRate != reqRate) Log.i(TAG, "ERROR rate $askedRate not supported, using 1")
        val reqPower = try { CameraPower.from(intent) } catch (e: IllegalArgumentException) {
            Log.i(TAG, "ERROR camera power options refused", e)
            stopSelf()
            return START_NOT_STICKY
        }
        if (!running) {
            power = reqPower
            running = true
            mode = reqMode
            rate = reqRate
            sessionId = newSessionId()
            synchronized(lock) {
                latest = null
                publishing = true
                deletePublished()
            }
            Log.i(TAG, "start: session $sessionId, mode $mode, rate $rate/s")
            camThread = HandlerThread("robotcam-camera").also { it.start() }
            camHandler = Handler(camThread.looper)
            camHandler.post { openCamera() }
            writer = Thread(::writeLoop, "robotcam-writer").also { it.start() }
        } else if (reqMode != mode || reqRate != rate || reqPower != power) {
            camHandler.post {
                Log.i(TAG, "switching mode $mode -> $reqMode, rate $rate -> $reqRate/s")
                power = reqPower
                mode = reqMode
                rate = reqRate
                restartCamera("config change")
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
    //
    // Every open attempt gets a generation number; closeCamera() starts a new one. A callback
    // that belongs to an older generation (an open, session or capture still in flight when a
    // mode/rate change, error or stop closed that attempt) closes what it was handed and
    // returns: it never touches the current camera and never unpublishes.

    private fun openCamera() {
        if (!running || device != null || opening) return
        val myGen = gen
        val manager = getSystemService(CameraManager::class.java)
        try {
            if (power.dump) {
                outDir.mkdirs()
                writeAtomic("characteristics.json", CameraPower.characteristics(manager).toByteArray())
            }
            val id = if (power.cameraId.isNotEmpty()) {
                require(power.cameraId in manager.cameraIdList) { "camera_id not enumerated" }
                require(manager.getCameraCharacteristics(power.cameraId).get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK) { "camera_id must face BACK" }
                power.cameraId
            } else manager.cameraIdList.first {
                manager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
            }
            val chars = manager.getCameraCharacteristics(id)
            power.configure(chars, id)
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
            yuvSizes = map.getOutputSizes(ImageFormat.YUV_420_888).toList()
            val (chosen, rule) = chooseFrameSize(yuvSizes)
            size = chosen
            sizeRule = rule
            fps = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)!!
                .minWith(compareBy<Range<Int>>({ it.upper }, { it.lower }))
            realtimeTimestamps = chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
            val sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
            exifOrientation = exifForRotation(sensorOrientation)
            Log.i(TAG, "camera $id YUV output sizes: ${yuvSizes.joinToString(" ")}")
            Log.i(TAG, "chosen ${size.width}x${size.height} ($rule), mode $mode, rate $rate/s, " +
                "fps $fps, sensor orientation $sensorOrientation (EXIF $exifOrientation), " +
                "realtime timestamps $realtimeTimestamps, attempt $myGen")

            frameReader = ImageReader.newInstance(size.width, size.height,
                ImageFormat.YUV_420_888, 3).apply {
                setOnImageAvailableListener({ r -> onFrame(r) }, camHandler)
            }
            if (mode == MODE_B && power.frameMs == 0) {
                val small = choosePreviewSize(yuvSizes, size)
                Log.i(TAG, "preview surface YUV ${small.width}x${small.height}")
                previewReader = ImageReader.newInstance(small.width, small.height,
                    ImageFormat.YUV_420_888, 2).apply {
                    setOnImageAvailableListener({ r -> onPreview(r) }, camHandler)
                }
            }
            opening = true
            manager.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    if (myGen != gen || !running) { camera.close(); return }
                    opening = false
                    device = camera
                    try {
                        startSession(camera, myGen)
                    } catch (e: Exception) {
                        failed("createCaptureSession", e)
                    }
                }
                override fun onDisconnected(camera: CameraDevice) = lost(camera, myGen, "disconnected")
                override fun onError(camera: CameraDevice, error: Int) = lost(camera, myGen, "error $error")
            }, camHandler)
        } catch (e: Exception) {
            // CameraAccessException, SecurityException, IllegalArgumentException, no back camera.
            failed("openCamera", e)
        }
    }

    private fun startSession(camera: CameraDevice, myGen: Long) {
        val frameSurface = frameReader!!.surface
        val previewSurface = previewReader?.surface
        val callback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) {
                if (myGen != gen || !running) { s.close(); return }
                session = s
                try {
                    nextStreamDue = 0L
                    if (power.frameMs != 0) {
                        // Opt-in manual B streams its chosen template directly to the frame
                        // surface. No still is queued behind long preview pipeline frames.
                        s.setRepeatingRequest(request(camera,
                            if (mode == MODE_B) power.stillTemplate else CameraDevice.TEMPLATE_PREVIEW,
                            frameSurface), resultCallback(myGen), camHandler)
                    } else if (mode == MODE_A) {
                        s.setRepeatingRequest(request(camera, CameraDevice.TEMPLATE_PREVIEW,
                            frameSurface), resultCallback(myGen), camHandler)
                    } else {
                        s.setRepeatingRequest(request(camera, CameraDevice.TEMPLATE_PREVIEW,
                            previewSurface!!), resultCallback(myGen), camHandler)
                        stillRequest = request(camera, power.stillTemplate,
                            frameSurface)
                        stillInFlight = false
                        camHandler.removeCallbacks(stillTick)
                        camHandler.postDelayed(stillTick, periodMs)
                    }
                    status("streaming")
                } catch (e: Exception) {
                    failed("start requests", e)
                }
            }
            override fun onConfigureFailed(s: CameraCaptureSession) {
                if (myGen != gen) { s.close(); return }
                failed("session configuration", null)
            }
        }
        val outputs = listOfNotNull(frameSurface, previewSurface).map { OutputConfiguration(it) }
        camera.createCaptureSession(SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR, outputs, Executor { camHandler.post(it) }, callback
        ))
    }

    private fun request(camera: CameraDevice, template: Int, target: android.view.Surface) =
        camera.createCaptureRequest(template).apply {
            addTarget(target)
            set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
            power.apply(this)
        }.build()

    private fun resultCallback(myGen: Long) = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult
        ) { observeResult(myGen, request, result) }
    }

    private fun observeResult(myGen: Long, request: CaptureRequest, result: TotalCaptureResult) {
        if (myGen != gen || !running) return
        try {
            if (power.observe(request, result)) {
                val camera = device ?: return
                val streamingFrames = mode == MODE_A || power.frameMs != 0
                val target = if (streamingFrames) frameReader!!.surface else previewReader!!.surface
                val template = if (mode == MODE_B && power.frameMs != 0) power.stillTemplate else CameraDevice.TEMPLATE_PREVIEW
                session!!.setRepeatingRequest(this.request(camera, template, target),
                    resultCallback(myGen), camHandler)
                if (mode == MODE_B && power.frameMs == 0) stillRequest = this.request(camera, power.stillTemplate, frameReader!!.surface)
            }
        } catch (e: Exception) {
            failed("power result/request", e)
        }
    }

    /** Mode B: one still per period; skipped while the previous one is still in flight. */
    private fun captureStill() {
        val s = session ?: return
        val req = stillRequest ?: return
        val myGen = gen
        camHandler.postDelayed(stillTick, periodMs)
        if (stillInFlight) return
        try {
            stillInFlight = true
            s.capture(req, object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult
                ) {
                    if (myGen != gen || !running) return
                    stillInFlight = false
                    observeResult(myGen, request, result)
                }
                override fun onCaptureFailed(
                    session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure
                ) {
                    if (myGen != gen) return
                    stillInFlight = false
                    Log.i(TAG, "ERROR still capture failed, reason ${failure.reason}")
                }
            }, camHandler)
        } catch (e: Exception) {
            failed("still capture", e)
        }
    }

    /**
     * Mode A receives every stream frame here and keeps one per period; mode B receives only
     * its stills. Only a frame that will be published is copied out of the camera buffer.
     * A callback from a reader that is no longer current is ignored; any failure while taking
     * the image goes through the normal unpublish-and-retry path.
     */
    private fun onFrame(r: ImageReader) {
        if (r !== frameReader) return
        try {
            val image = r.acquireLatestImage() ?: return
            image.use {
                if (power.frameMs != 0) {
                    val now = if (realtimeTimestamps) it.timestamp / 1_000_000 else SystemClock.elapsedRealtime()
                    // Anchor to the prior deadline: now+period skips every other 1 Hz
                    // frame when callbacks jitter. Half a frame selects the nearest sample.
                    val tolerance = minOf(power.frameMs.toLong(), periodMs) / 2
                    if (now < nextStreamDue - tolerance) return
                    nextStreamDue = if (nextStreamDue == 0L) now + periodMs else
                        maxOf(nextStreamDue + periodMs, now + periodMs - tolerance)
                } else if (mode == MODE_A) {
                    val now = SystemClock.elapsedRealtime()
                    if (now < nextStreamDue) return
                    nextStreamDue = now + periodMs
                }
                val nv21 = toNv21(it)
                // Boot clock (elapsedRealtime, CLOCK_BOOTTIME): the sensor timestamp when it is
                // on that base, otherwise the arrival time. Wall clock derived from it for humans.
                val bootNowNs = SystemClock.elapsedRealtimeNanos()
                val bootNs = if (realtimeTimestamps) it.timestamp else bootNowNs
                val bootMs = bootNs / 1_000_000
                val wallMs = System.currentTimeMillis() - (bootNowNs - bootNs) / 1_000_000
                synchronized(lock) {
                    if (!publishing) return
                    latest = Frame(nv21, it.width, it.height, bootMs, wallMs, ++cameraSeq)
                    lock.notifyAll()
                }
            }
        } catch (e: Exception) {
            // IllegalStateException (reader closed, too many images), buffer errors.
            failed("frame read", e)
        }
    }

    /** Mode B preview frames only keep 3A running; drop them. */
    private fun onPreview(r: ImageReader) {
        if (r !== previewReader) return
        try {
            r.acquireLatestImage()?.close()
        } catch (e: Exception) {
            failed("preview read", e)
        }
    }

    private fun lost(camera: CameraDevice, myGen: Long, why: String) {
        camera.close()
        if (myGen != gen) return
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

    /** Releases the current attempt and makes every callback still in flight for it obsolete. */
    private fun closeCamera() {
        gen++
        opening = false
        camHandler.removeCallbacks(stillTick)
        stillRequest = null
        stillInFlight = false
        try { session?.close() } catch (e: IllegalStateException) { /* already closed */ }
        session = null
        device?.close(); device = null
        frameReader?.close(); frameReader = null
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
        while (true) {
            // Wait for a frame newer than the last one published. Pacing is done on camThread.
            val f = synchronized(lock) {
                while (true) {
                    if (!publishing) return
                    val cur = latest
                    if (cur != null && cur.seq != lastSeq) break
                    try { lock.wait() } catch (e: InterruptedException) { return }
                }
                latest!!
            }
            lastSeq = f.seq
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
        val t0 = SystemClock.elapsedRealtime()
        val raw = ByteArrayOutputStream(64 * 1024)
        YuvImage(f.nv21, ImageFormat.NV21, f.width, f.height, null)
            .compressToJpeg(Rect(0, 0, f.width, f.height), JPEG_QUALITY, raw)
        val encodeMs = SystemClock.elapsedRealtime() - t0
        val clock = if (realtimeTimestamps) "sensor" else "arrival"
        val jpeg = withComment(withExifOrientation(raw.toByteArray(), exifOrientation),
            "robotcam session=$sessionId frame=$n capture_boot_ms=${f.captureBootMs} " +
                "capture_wall_ms=${f.captureWallMs} clock=$clock")
        val json = """{"session_id":"$sessionId","frame":$n,"capture_boot_ms":${f.captureBootMs},""" +
            """"capture_wall_ms":${f.captureWallMs},""" +
            """"written_wall_ms":${System.currentTimeMillis()},"mode":"$mode","rate":$rate,""" +
            """"width":${f.width},"height":${f.height},"size_rule":"$sizeRule",""" +
            """"jpeg_quality":$JPEG_QUALITY,"encode_ms":$encodeMs,"exif_orientation":$exifOrientation,""" +
            """"bytes":${jpeg.size},"fps_min":${fps.lower},"fps_max":${fps.upper},""" +
            """"clock":"$clock","variant":${org.json.JSONObject.quote(power.variant)},"power":${power.diagnostics()},""" +
            """"yuv_sizes":[${yuvSizes.joinToString(",") { "\"${it.width}x${it.height}\"" }}]}""" +
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
        // Heartbeat, so a look at logcat always finds a recent line.
        if (n == 1L || n % HEARTBEAT_FRAMES == 0L) {
            Log.i(TAG, "published frame $n, ${f.width}x${f.height}, mode $mode, rate $rate/s, " +
                "encode $encodeMs ms, ${jpeg.size} B, write errors $writeErrors")
        }
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
            append("mode $mode, $rate/s, ${size.width}x${size.height}, fps ${fps.lower}-${fps.upper}, ")
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
        private const val JPEG_QUALITY = 85
        private const val REOPEN_DELAY_MS = 2000L
        private const val HEARTBEAT_FRAMES = 30L
        private const val JPEG_NAME = "frame.jpg"
        private const val SIDECAR_NAME = "frame.json"
        const val EXTRA_MODE = "mode"
        const val EXTRA_RATE = "rate"
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
        fun chooseFrameSize(sizes: List<Size>): Pair<Size, String> {
            sizes.firstOrNull { it.width == 640 && it.height == 480 }?.let { return it to "exact 640x480" }
            val byArea = sizes.sortedBy { it.width.toLong() * it.height }
            byArea.firstOrNull { it.width * 3 == it.height * 4 && it.width >= 640 && it.height >= 480 }
                ?.let { return it to "smallest 4:3 >= 640x480" }
            byArea.firstOrNull { it.width >= 640 }?.let { return it to "smallest >= 640 wide" }
            return byArea.last() to "largest (none >= 640 wide)"
        }

        /** Smallest preview size with the frame's aspect ratio (same field of view), else smallest. */
        fun choosePreviewSize(sizes: List<Size>, frame: Size): Size {
            val byArea = sizes.sortedBy { it.width.toLong() * it.height }
            return byArea.firstOrNull {
                it.width.toLong() * frame.height == it.height.toLong() * frame.width && it.width >= 320
            } ?: byArea.first()
        }

        /** EXIF orientation that tells a viewer to rotate the image clockwise by [degrees]. */
        fun exifForRotation(degrees: Int) = when (degrees) {
            90 -> 6
            180 -> 3
            270 -> 8
            else -> 1
        }

        /** Copies a YUV_420_888 image into an NV21 array (Y plane, then interleaved V/U). */
        private fun toNv21(image: Image): ByteArray {
            val w = image.width
            val h = image.height
            val out = ByteArray(w * h * 3 / 2)
            val y = image.planes[0]
            val yBuf = y.buffer
            for (row in 0 until h) {
                yBuf.position(row * y.rowStride)
                yBuf.get(out, row * w, w)
            }
            val u = image.planes[1]
            val v = image.planes[2]
            val uBuf = u.buffer
            val vBuf = v.buffer
            var o = w * h
            for (row in 0 until h / 2) {
                for (col in 0 until w / 2) {
                    out[o++] = vBuf.get(row * v.rowStride + col * v.pixelStride)
                    out[o++] = uBuf.get(row * u.rowStride + col * u.pixelStride)
                }
            }
            return out
        }

        /**
         * Inserts a minimal EXIF APP1 segment holding only the Orientation tag, right after SOI.
         * The in-app encoder writes no EXIF; readers such as PIL's exif_transpose use this tag.
         */
        fun withExifOrientation(jpeg: ByteArray, orientation: Int): ByteArray {
            if (jpeg.size < 2 || jpeg[0] != 0xFF.toByte() || jpeg[1] != 0xD8.toByte()) return jpeg
            val tiff = byteArrayOf(
                'M'.code.toByte(), 'M'.code.toByte(), 0, 42, 0, 0, 0, 8, // big-endian, IFD0 at 8
                0, 1,                                                  // one entry
                0x01, 0x12, 0, 3, 0, 0, 0, 1,                          // Orientation, SHORT, count 1
                0, orientation.toByte(), 0, 0,                         // value, padding
                0, 0, 0, 0                                             // no next IFD
            )
            val body = "Exif".toByteArray(Charsets.US_ASCII) + byteArrayOf(0, 0) + tiff
            val len = body.size + 2
            val segment = byteArrayOf(0xFF.toByte(), 0xE1.toByte(), (len shr 8).toByte(),
                (len and 0xFF).toByte()) + body
            return jpeg.copyOfRange(0, 2) + segment + jpeg.copyOfRange(2, jpeg.size)
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
