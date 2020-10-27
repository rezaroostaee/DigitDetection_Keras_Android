package ir.rezaroostaee.www.digitdetection.utils;
import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.util.Log;



import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class Recognizer {
    private Executor executor = Executors.newSingleThreadExecutor();
    private Classifier classifier;

    private static final String TAG = "MainActivity";

    private static final int PIXEL_WIDTH = 28;

    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "input_input";
    private static final String OUTPUT_NAME = "output/Softmax";

//    private static final String INPUT_NAME = "input";
//    private static final String OUTPUT_NAME = "output";

    String modelPath;

    String labelPath;
    private Activity parentActivity;

    public Recognizer(Activity parentActivity, String modelPath, String labelPath) {
        this.modelPath = modelPath;
        this.labelPath = labelPath;
        this.parentActivity = parentActivity;
        initTensorFlowAndLoadModel();
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            parentActivity.getAssets(),
                            modelPath,
                            labelPath,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private Bitmap optimizedCrop(Bitmap bm) {
        int maxLeftBound = bm.getWidth() * 3 / 8;
//        int avgColor = bm.getPixel(0, bm.getHeight() / 2);
//        avgColor = (Color.red(avgColor) + Color.green(avgColor) + Color.blue(avgColor)) / 3;
        int leftBound = bm.getWidth() * 15 / 80;

        int pixelCount = 0;
        int grayBucket = 0;
        int minColor = 300;
        int maxColor = 0;

        for (int y = 0; y < bm.getHeight(); y++) {
            for (int x = 0; x < bm.getWidth(); x++) {
                int clr = bm.getPixel(x, y);
                clr = (Color.red(clr) + Color.green(clr) + Color.blue(clr)) / 3;

                pixelCount++;
                grayBucket += clr;

                if (clr < minColor) {
                    minColor = clr;
                }
                if (clr > maxColor){
                    maxColor = clr;
                }
            }
        }

        int avgGray = grayBucket / pixelCount;
        int newDarkColor = (avgGray + minColor) / 2;

        for (int y = 0; y < bm.getHeight(); y++) {
            for (int x = 0; x < bm.getWidth(); x++) {
                int clr = bm.getPixel(x, y);
                clr = (Color.red(clr) + Color.green(clr) + Color.blue(clr)) / 3;

                if (clr < avgGray) {
                    break;
                }
                bm.setPixel(x, y, Color.rgb(newDarkColor, newDarkColor, newDarkColor));
            }
        }

        for (int y = bm.getHeight() - 1; y > 0; y--) {
            for (int x = 0; x < bm.getWidth(); x++) {
                int clr = bm.getPixel(x, y);
                clr = (Color.red(clr) + Color.green(clr) + Color.blue(clr)) / 3;

                if (clr < avgGray) {
                    break;
                }
                bm.setPixel(x, y, Color.rgb(newDarkColor, newDarkColor, newDarkColor));
            }
        }


        bm = Bitmap.createBitmap(bm, leftBound, bm.getHeight() / 11,
                bm.getWidth() - (bm.getWidth() * 15 / 80 + leftBound), bm.getHeight() * 9 / 11);

//        bm = Bitmap.createBitmap(bm, bm.getWidth() * 15 / 80, bm.getHeight() / 9,
//                bm.getWidth() * 5 / 8, bm.getHeight() * 7 / 9);
        return bm;
    }

    public String recognize(Bitmap bm) {
        //bm = optimizedCrop(bm);
        float pixels[] = getPixelData(bm);

        int p[] = new int[784];
        for (int i = 0; i < p.length; i++) {
            p[i] = Color.rgb((int) pixels[i], (int) pixels[i], (int) pixels[i]);
        }

//        Bitmap bitmap = Bitmap.createBitmap(PIXEL_WIDTH, PIXEL_WIDTH, Bitmap.Config.ARGB_8888);
//        bitmap.setPixels(p, 0, PIXEL_WIDTH, 0,0,PIXEL_WIDTH,PIXEL_WIDTH);


        final List<Classifier.Recognition> results = classifier.recognizeImage(pixels);

        return results.get(0).getTitle();
    }

    public float[] getPixelData(Bitmap mOffscreenBitmap) {
        mOffscreenBitmap = Bitmap.createScaledBitmap(mOffscreenBitmap, PIXEL_WIDTH, PIXEL_WIDTH, true);
//        mOffscreenBitmap = toGrayScale(mOffscreenBitmap);

        if (mOffscreenBitmap == null) {
            return null;
        }

        int width = mOffscreenBitmap.getWidth();
        int height = mOffscreenBitmap.getHeight();

        // Get 32x32 pixel data from bitmap
        int[] pixels = new int[width * height];
        mOffscreenBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] retPixels = new float[pixels.length];

        for (int i = 0; i < pixels.length; i++) {
            // Set 0 for white and 1 for black pixel
            int gray = (Color.red(pixels[i]) + Color.blue(pixels[i]) + Color.green(pixels[i])) / 3;
            retPixels[i] = gray;
        }

        return retPixels;
    }

    public Bitmap toGrayScale(Bitmap bmpOriginal) {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    public float[] histogramEqualization(float[] pixel) {
        int anzpixel = pixel.length;
        int[] histogram = new int[255];
        int[] iarray = new int[1];
        int i = 0;

        //read pixel intensities into histogram
        for (float valueBefore : pixel) {
            histogram[(int) valueBefore]++;
        }

        int sum = 0;
        // build a Lookup table LUT containing scale factor
        float[] lut = new float[anzpixel];
        for (i = 0; i < 255; ++i) {
            sum += histogram[i];
            lut[i] = sum * 255 / anzpixel;
        }

        // transform image using sum histogram as a Lookup table
        for (int x = 0; x < anzpixel; x++) {
            float valueBefore = pixel[x];
            int valueAfter = (int) lut[(int) valueBefore];
            pixel[x] = valueAfter;
        }
        return pixel;
    }
}
