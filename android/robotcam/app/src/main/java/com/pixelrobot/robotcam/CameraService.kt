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
import android.hardware.camera2.CaptureRequest
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
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs

/**
 * Keeps the back camera open, streaming small JPEGs at the lowest supported frame rate,
 * holds only the newest frame in memory, and about once a second writes it to
 * Download/robotcam/frame.jpg plus a frame.json sidecar, each via temp file + rename.
 */
class CameraService : Service() {

    private class Frame(val jpeg: ByteArray, val captureWallMs: Long, val cameraSeq: Long)

    private val latest = AtomicReference<Frame?>(null)

    @Volatile private var running = false
    private val sessionStartMs = System.currentTimeMillis()

    // Camera state; touched only on camThread.
    private lateinit var camThread: HandlerThread
    private lateinit var camHandler: Handler
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var cameraSeq = 0L
    private val reopen = Runnable { openCamera() }

    // Chosen configuration, reported in the notification and the sidecar.
    @Volatile private var size = Size(0, 0)
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
            Log.e(TAG, "startForeground refused", e)
            stopSelf()
            return START_NOT_STICKY
        } catch (e: IllegalStateException) {
            Log.e(TAG, "startForeground refused", e)
            stopSelf()
            return START_NOT_STICKY
        }
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "CAMERA permission not granted; open the RobotCam app once to grant it")
            stopSelf()
            return START_NOT_STICKY
        }
        if (!running) {
            running = true
            camThread = HandlerThread("robotcam-camera").also { it.start() }
            camHandler = Handler(camThread.looper)
            camHandler.post { openCamera() }
            writer = Thread(::writeLoop, "robotcam-writer").also { it.start() }
        }
        // Not sticky: a restart by the system would come from the background, where the
        // camera is not available to a foreground service anyway.
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        if (running) {
            running = false
            writer?.interrupt()
            writer?.join(2000)
            camHandler.post { closeCamera() }
            camThread.quitSafely()
            // The camera is off: remove the files so no reader mistakes the last frame for a live one.
            File(outDir, JPEG_NAME).delete()
            File(outDir, SIDECAR_NAME).delete()
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
            size = map.getOutputSizes(ImageFormat.JPEG)
                .minBy { abs(it.width - WANT_W) + abs(it.height - WANT_H) }
            fps = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)!!
                .minWith(compareBy<Range<Int>>({ it.upper }, { it.lower }))
            realtimeTimestamps = chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
                CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
            val orientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
            Log.i(TAG, "camera $id: jpeg ${size.width}x${size.height} (wanted ${WANT_W}x$WANT_H), " +
                "fps $fps, sensor orientation $orientation, realtime timestamps $realtimeTimestamps")

            reader = ImageReader.newInstance(size.width, size.height, ImageFormat.JPEG, 2).apply {
                setOnImageAvailableListener({ r -> onImage(r) }, camHandler)
            }
            manager.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    if (!running) { camera.close(); return }
                    device = camera
                    startSession(camera, orientation)
                }
                override fun onDisconnected(camera: CameraDevice) = lost(camera, "disconnected")
                override fun onError(camera: CameraDevice, error: Int) = lost(camera, "error $error")
            }, camHandler)
        } catch (e: Exception) {
            // CameraAccessException, SecurityException, or no back camera: report and retry.
            Log.e(TAG, "openCamera failed", e)
            status("camera open failed: ${e.message}")
            closeCamera()
            scheduleReopen()
        }
    }

    private fun startSession(camera: CameraDevice, orientation: Int) {
        val surface = reader!!.surface
        val callback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) {
                if (!running) { s.close(); return }
                session = s
                val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(surface)
                    set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps)
                    set(CaptureRequest.JPEG_QUALITY, JPEG_QUALITY)
                    // Same as termux-camera-photo with the phone upright.
                    set(CaptureRequest.JPEG_ORIENTATION, orientation)
                }.build()
                s.setRepeatingRequest(request, null, camHandler)
                status("streaming")
            }
            override fun onConfigureFailed(s: CameraCaptureSession) {
                Log.e(TAG, "capture session configuration failed")
                status("session configuration failed")
                closeCamera()
                scheduleReopen()
            }
        }
        camera.createCaptureSession(SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR,
            listOf(OutputConfiguration(surface)),
            Executor { camHandler.post(it) },
            callback
        ))
    }

    private fun onImage(r: ImageReader) {
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
            latest.set(Frame(bytes, wall, ++cameraSeq))
        }
    }

    private fun lost(camera: CameraDevice, why: String) {
        Log.w(TAG, "camera $why")
        status("camera $why, retrying")
        camera.close()
        if (device === camera) device = null
        closeCamera()
        scheduleReopen()
    }

    private fun scheduleReopen() {
        camHandler.removeCallbacks(reopen)
        if (running) camHandler.postDelayed(reopen, REOPEN_DELAY_MS)
    }

    private fun closeCamera() {
        session?.close(); session = null
        device?.close(); device = null
        reader?.close(); reader = null
    }

    // ---------------------------------------------------------------- writer thread

    private fun writeLoop() {
        var next = SystemClock.elapsedRealtime() + WRITE_PERIOD_MS
        var lastSeq = 0L
        while (running) {
            try {
                Thread.sleep(maxOf(0L, next - SystemClock.elapsedRealtime()))
            } catch (e: InterruptedException) {
                return
            }
            next += WRITE_PERIOD_MS
            val f = latest.get()
            // No new camera frame since the last write: leave the files alone. The reader sees an
            // unchanged frame counter and an ageing capture time.
            if (f == null || f.cameraSeq == lastSeq) continue
            lastSeq = f.cameraSeq
            try {
                writeFrame(f)
            } catch (e: IOException) {
                writeFailed(e)
            } catch (e: ErrnoException) {
                writeFailed(e)
            }
        }
    }

    private fun writeFrame(f: Frame) {
        outDir.mkdirs()
        val n = framesWritten + 1
        val jpeg = withComment(f.jpeg, "robotcam frame=$n capture_wall_ms=${f.captureWallMs}")
        val json = """{"frame":$n,"capture_wall_ms":${f.captureWallMs},""" +
            """"written_wall_ms":${System.currentTimeMillis()},"session_start_ms":$sessionStartMs,""" +
            """"width":${size.width},"height":${size.height},"bytes":${jpeg.size},""" +
            """"fps_min":${fps.lower},"fps_max":${fps.upper},""" +
            """"timestamp_source":"${if (realtimeTimestamps) "sensor" else "arrival"}"}""" + "\n"
        // Image first, sidecar last: a sidecar always describes an image that is already in place.
        writeAtomic(JPEG_NAME, jpeg)
        writeAtomic(SIDECAR_NAME, json.toByteArray())
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
        Log.e(TAG, "write failed", e)
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
            append("${size.width}x${size.height}, fps ${fps.lower}-${fps.upper}, ")
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
        private const val WANT_W = 640
        private const val WANT_H = 480
        private const val JPEG_QUALITY: Byte = 85
        private const val WRITE_PERIOD_MS = 1000L
        private const val REOPEN_DELAY_MS = 2000L
        private const val JPEG_NAME = "frame.jpg"
        private const val SIDECAR_NAME = "frame.json"

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
