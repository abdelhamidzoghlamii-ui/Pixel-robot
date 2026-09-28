package com.pixelrobot.robotcam

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle

/**
 * Invisible activity: asks for the camera (and, on Android 13+, notification) permission
 * once, then starts [CameraService] and finishes.
 *
 * The service is started from an activity rather than from a broadcast because a
 * camera-type foreground service may only use the camera if it was started while the
 * app was in the foreground.
 */
class StartActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val wanted = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= 33) wanted += Manifest.permission.POST_NOTIFICATIONS
        val missing = wanted.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) {
            startAndFinish()
        } else {
            requestPermissions(missing.toTypedArray(), 1)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        startAndFinish()
    }

    private fun startAndFinish() {
        // Without the camera permission the service stops itself and says why in logcat.
        // `--es mode A` selects the repeating JPEG stream; default is mode B (preview + stills).
        startForegroundService(Intent(this, CameraService::class.java)
            .putExtra(CameraService.EXTRA_MODE, intent.getStringExtra(CameraService.EXTRA_MODE)))
        finish()
    }
}
