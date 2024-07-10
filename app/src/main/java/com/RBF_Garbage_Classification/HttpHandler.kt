package com.RBF_Garbage_Classification

import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import org.json.JSONObject
import java.io.IOException
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.MalformedURLException
import java.net.ProtocolException
import java.net.URL
import java.util.Base64

class HttpHandler {
    @RequiresApi(Build.VERSION_CODES.O)
    fun uploadImage(requestUrl: String, imageByteArray: ByteArray): Pair<Boolean, String?> {
        val imageBase64: String = Base64.getEncoder().encodeToString(imageByteArray)
        val jsonObject = JSONObject()
        jsonObject.put("image", imageBase64)

        try {
            val url = URL(requestUrl)
            val conn: HttpURLConnection = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.doInput = true
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json")
            conn.setRequestProperty("Accept", "application/json")

            val outputStream = OutputStreamWriter(conn.outputStream)
            outputStream.write(jsonObject.toString())
            outputStream.flush()

            val responseCode = conn.responseCode
            val responseMessage = conn.responseMessage
            val result = if (responseCode == HttpURLConnection.HTTP_OK) {
                val responseStream = conn.inputStream.bufferedReader().use { it.readText() }
                val responseJson = JSONObject(responseStream)
                val classification = responseJson.optString("classification", null)
                classification
            } else {
                null
            }
            conn.disconnect()
            return Pair(responseCode == HttpURLConnection.HTTP_OK, result)

        } catch (ex: MalformedURLException) {
            Log.e("HttpHandler", "MalformedURLException: " + ex.message)
            return Pair(false, null)
        } catch (ex: ProtocolException) {
            Log.e("HttpHandler", "ProtocolException: " + ex.message)
            return Pair(false, null)
        } catch (ex: IOException) {
            Log.e("HttpHandler", "IOException: " + ex.message)
            return Pair(false, null)
        } catch (ex: Exception) {
            Log.e("HttpHandler", "Exception: " + ex.message)
            return Pair(false, null)
        }
    }
}