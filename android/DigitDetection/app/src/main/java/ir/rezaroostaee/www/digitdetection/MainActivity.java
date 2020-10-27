package ir.rezaroostaee.www.digitdetection;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.FileNotFoundException;
import java.io.InputStream;

import ir.rezaroostaee.www.digitdetection.utils.Recognizer;

public class MainActivity extends AppCompatActivity {

    Recognizer digitRecognizer;

    int RESULT_LOAD_IMG = 12345;

    Bitmap bmp = null;

    ImageView img;
    TextView result;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        iniUI();
        loadDigitRecognitionModel();

    }

    private void iniUI(){
        img = (ImageView) findViewById(R.id.image);
        result = (TextView) findViewById(R.id.result);
    }

    private void loadDigitRecognitionModel() {
        digitRecognizer = new Recognizer(this,
                "file:///android_asset/recognition_model.pb",
                "file:///android_asset/graph_label_strings.txt");
    }

    public void predictDigit(){
        if (bmp == null){
            Toast.makeText(this, "Select one picture.", Toast.LENGTH_LONG).show();
            return;
        }
        String digitPrd = digitRecognizer.recognize(bmp);
        result.setText("predicted digit is : " + digitPrd);
    }

    public void selectFromGallery(View v){
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
        photoPickerIntent.setType("image/*");
        startActivityForResult(photoPickerIntent, RESULT_LOAD_IMG);
    }

    @Override
    protected void onActivityResult(int reqCode, int resultCode, Intent data) {
        super.onActivityResult(reqCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            try {
                final Uri imageUri = data.getData();
                final InputStream imageStream = getContentResolver().openInputStream(imageUri);
                bmp = BitmapFactory.decodeStream(imageStream);
                img.setImageBitmap(bmp);
                predictDigit();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG).show();
            }

        }else {
            Toast.makeText(this, "You haven't picked Image",Toast.LENGTH_LONG).show();
        }
    }
}
