package com.pixelrobot.robotcam

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Stops [CameraService] on an explicit `com.pixelrobot.robotcam.STOP` broadcast. */
class ControlReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == ACTION_STOP) {
            context.stopService(Intent(context, CameraService::class.java))
        }
    }

    companion object {
        const val ACTION_STOP = "com.pixelrobot.robotcam.STOP"
    }
}
