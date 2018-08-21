package ai.treeleaf.signature;

import org.omg.CORBA.MARSHAL;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.videoio.VideoCapture;

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
        filename = "data/image.jpg";
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
//        HighGui.waitKey(0);


//        Mat img4 = img3.reshape((int) (size.width*size.height),1);
//        System.out.println(img4.rows() + "x" + img4.cols() + "x" + img4.channels());
//        System.out.println(img4.elemSize());
        double [] feature = img3.reshape((int) (size.width*size.height),1).get(0,0);
        System.out.println("Features: "+Arrays.toString(feature));
        return feature;
    }

    public static void main(String[] args) throws InterruptedException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat mat = Mat.ones(3, 3, CvType.CV_8UC1);
        Mat mat1 = Mat.eye(3, 3, CvType.CV_8UC1);
        System.out.println("mat = " + mat.dump());
        System.out.println("mat1 = " + mat1.dump());
        Mat matAdd = new Mat();
        Core.add(mat, mat1, matAdd);
        System.out.println("mat + mat1 = " + matAdd.dump());

        String filename = "data/image.jpg";
        Size size = new Size(128,128);
//        Mat img = Imgcodecs.imread(filename);
//        System.out.println("Input image: " + filename);
//        System.out.println(img.rows() + "x" + img.cols() + "x" + img.channels());
//        HighGui.namedWindow("Image");
//        HighGui.imshow("Image", img);
//        HighGui.waitKey(0);
////        System.out.println(img.dump());
////        for (int i = 0; i < img.rows(); i++) {
////            for (int j = 0; j < img.cols(); j++) {
////                    System.out.println(Arrays.toString(img.get(i, j)));
////            }
////        }
//
//        Mat img1 = new Mat();
//        Imgproc.resize(img, img1, size);
//        System.out.println(img1.rows() + "x" + img1.cols() + "x" + img1.channels());
//        HighGui.imshow("Image", img1);
//        HighGui.waitKey(0);
//
//        Mat img2 = new Mat();
//        Imgproc.cvtColor(img1, img2, Imgproc.COLOR_RGB2GRAY);
//        System.out.println(img2.rows() + "x" + img2.cols() + "x" + img2.channels());
//        HighGui.imshow("Image", img2);
//        HighGui.waitKey(0);
//
//        Mat img4 = new Mat();
//        Imgproc.threshold(img2, img4, 100, 255, Imgproc.THRESH_BINARY);
//        HighGui.imshow("Image", img4);
//        HighGui.waitKey(0);
//
//
//        Mat img3 = img4.reshape((int) (size.width*size.height),1);
//        System.out.println(img3.rows() + "x" + img3.cols() + "x" + img3.channels());
//        System.out.println(img3.elemSize());
//        double [] feature = img2.reshape((int) (size.width*size.height),1).get(0,0);
//        System.out.println("Features: "+Arrays.toString(feature));


//        List<double[]> features = new ArrayList<double[]>();
//        List<Integer> targets = new ArrayList<Integer>();
//
//        for all image in folder{
//            String f = "";
//            int target = 0;
//            double[] feature = computeFeatures(f);
//            features.add(feature);
//            targets.add(target);
//        }
        // Read file images.txt
        // File format
        // filename1 0
        // filename2 0
        // filename3 0
        // filename4 1
        // filename5 2
        List<String[]> files = new ArrayList<String[]>();
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","0"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","0"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","0"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","1"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","1"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","1"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","2"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","2"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","2"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","1"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","0"});
        files.add(new String[]{"/home/ashok/Projects/ashok/signature/data/signature_dataset/person1/image1.png","2"});

        files.forEach(e->{
            System.out.println(e[0]+" "+e[1]);
        });


        int numSamples = files.size();
        int numFeatures = (int) (size.height*size.width);
        double[][] features = new double[numSamples][numFeatures];
        int[] targets = new int[numSamples];

        for(int i = 0; i<numSamples; i++){
            String f = files.get(i)[0];
            int target = Integer.parseInt(files.get(i)[1]);

            double[] feat = computeFeatures(f, size);
            features[i] = feat;
            targets[i] =target;
        }

        System.out.println(Arrays.deepToString(features));
        System.out.println(Arrays.toString(targets));

        System.out.println("Training");
        LearnerPredictor learnerPredictor = new LearnerPredictor();
        learnerPredictor.fit(features, targets, features, targets);
        System.out.println("Training done.");
        for (int i = 0; i<features.length; i++){
            int output = learnerPredictor.predict(features[i]);
            int target = targets[i];
            System.out.println(output+ " "+ target);
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
