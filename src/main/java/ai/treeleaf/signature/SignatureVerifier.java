package ai.treeleaf.signature;

import org.omg.CORBA.MARSHAL;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.videoio.VideoCapture;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Ashok K. Pant
 * Email:(asokpant@gmail.com)
 * Date: 8/21/18
 */
public class SignatureVerifier {
    static {
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
    }

    public static double[] computeFeatures(String filename, Size size){
//        filename = "data/image.jpg";
        Mat img = Imgcodecs.imread(filename);
//        System.out.println("Input image: " + filename);
//        System.out.println(img.rows() + "x" + img.cols() + "x" + img.channels());
//        HighGui.namedWindow("Image");
//        HighGui.imshow("Image", img);
//        HighGui.waitKey(0);
//        System.out.println(img.dump());
//        for (int i = 0; i < img.rows(); i++) {
//            for (int j = 0; j < img.cols(); j++) {
//                    System.out.println(Arrays.toString(img.get(i, j)));
//            }
//        }

        Mat img1 = new Mat();
        Imgproc.resize(img, img1, size);
//        System.out.println(img1.rows() + "x" + img1.cols() + "x" + img1.channels());
//        HighGui.imshow("Image", img1);
//        HighGui.waitKey(0);

        Mat img2 = new Mat();
        Imgproc.cvtColor(img1, img2, Imgproc.COLOR_RGB2GRAY);
//        System.out.println(img2.rows() + "x" + img2.cols() + "x" + img2.channels());
//        HighGui.imshow("Image", img2);
//        HighGui.waitKey(0);

        Mat img3 = new Mat();
        Imgproc.threshold(img2, img3, 100, 255, Imgproc.THRESH_BINARY);
//        HighGui.imshow("Image", img3);
//        HighGui.waitKey(100);
        Core.bitwise_not(img3, img3);
//        HighGui.imshow("Image", img3);
//        HighGui.waitKey(100);
        Core.divide(img3,new Scalar(255), img3);


//        Mat img4 = img3.reshape((int) (size.width*size.height),1);
//        System.out.println(img4.rows() + "x" + img4.cols() + "x" + img4.channels());
//        System.out.println(img4.elemSize());
        Mat a = img3.reshape(1,1);
        double [] feature = new double[a.cols()];
        for (int i =0; i <feature.length; i++){
            feature[i] = a.get(0,i)[0];
        }
//        System.out.println("Features: "+Arrays.toString(feature));
        return feature;
    }

    public static List<String[]> read_files(String filename){

        List<String[]> files = new ArrayList<>();
        try {
            List<String> temp = Files.readAllLines(Paths.get(filename));
            temp.forEach(line->
                    files.add(line.split(" ")));

        } catch (IOException e) {
            e.printStackTrace();
        }
        return files;
    }
    public static void main(String[] args) throws InterruptedException {
        Size size = new Size(128,128);

        String datasetPath="data/signature_dataset1.txt";
        List<String[]> files = read_files(datasetPath);

//        files.forEach(e-> System.out.println(e[0]+" "+e[1]));


        int numSamples = files.size();
        int numFeatures = (int) (size.height*size.width);
        double[][] features = new double[numSamples][numFeatures];
        int[] targets = new int[numSamples];

        for(int i = 0; i<numSamples; i++){
            String f = files.get(i)[0];
            int target = Integer.parseInt(files.get(i)[1]);

            double[] feat = computeFeatures(f, size);
//            System.out.println(feat.length +" "+numFeatures);
            features[i] = feat;
            targets[i] =target;
        }

        System.out.println(Arrays.deepToString(features));
        System.out.println(Arrays.toString(targets));

        System.out.println("Training");
        LearnerPredictor learnerPredictor = new LearnerPredictor();
        learnerPredictor.fit(features, targets, features, targets);
        System.out.println("Training done.");

//        Evaluate model (precision, recall, fscore)

        for (int i = 0; i<features.length; i++){
            int output = learnerPredictor.predict(features[i]);
            int target = targets[i];
            System.out.println(target+ " "+ output);
        }

        VideoCapture videoCapture = new VideoCapture(0);
        HighGui.namedWindow("Video");

        while (true){
            Mat frame = new Mat();
            videoCapture.read(frame);

            HighGui.imshow("Video", frame);
            HighGui.waitKey(20);
        }
    }

}
