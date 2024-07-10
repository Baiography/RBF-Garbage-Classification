package com.RBF_Garbage_Classification

import android.graphics.Bitmap

interface MainActivityCallback {
    fun updateImageView(imageBitmap : Bitmap)
    fun showUploadButton()
    fun updateClassificationResult(result: String)
}