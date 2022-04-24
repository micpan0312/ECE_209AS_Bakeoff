//import java.io.File;
//import java.io.FileInputStream;
//import java.io.FileNotFoundException;
//import java.io.FileOutputStream;
//import java.io.IOException;
//import java.io.ObjectInputStream;
//import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.*;  
//import java.io.Serializable;

import processing.core.PApplet;
import processing.sound.AudioIn;
import processing.sound.FFT;
import processing.sound.Sound;
import processing.sound.Waveform;
import weka.core.SerializationHelper;

/* A class with the main function and Processing visualizations to run the demo */

public class ClassifyVibration extends PApplet {

	FFT fft;
	AudioIn in;
	Waveform waveform;
	int bands = 512;
	int nsamples = 1024;
	float[] spectrum = new float[bands];
	float[] fftFeatures = new float[bands];
//	String[] classNames = {"quiet", "hand drill", "whistling", "class clapping"};
	String[] classNames = {"neutral", "scenario 1", "scenario 2"};
	int classIndex = 0;
	int dataCount = 0;
	String cwd = Path.of("").toAbsolutePath().toString();
//	Date time = new java.util.Date();


	MLClassifier classifier;
	
	Map<String, List<DataInstance>> trainingData = new HashMap<>();
	{for (String className : classNames){
		trainingData.put(className, new ArrayList<DataInstance>());
	}}
	
	DataInstance captureInstance (String label){
		DataInstance res = new DataInstance();
		res.label = label;
		res.measurements = fftFeatures.clone();
		return res;
	}
	
	public static void main(String[] args) {
		PApplet.main("ClassifyVibration");
	}
	
	public void settings() {
		size(512, 400);
	}

	public void setup() {
		
		/* list all audio devices */
		Sound.list();
		Sound s = new Sound(this);
		  
		/* select microphone device */
		s.inputDevice(10);
		    
		/* create an Input stream which is routed into the FFT analyzer */
		fft = new FFT(this, bands);
		in = new AudioIn(this, 0);
		waveform = new Waveform(this, nsamples);
		waveform.input(in);
		
		/* start the Audio Input */
		in.start();
		  
		/* patch the AudioIn */
		fft.input(in);
	}

	public void draw() {
		background(0);
		fill(0);
		stroke(255);
		
		waveform.analyze();

		beginShape();
		  
		for(int i = 0; i < nsamples; i++)
		{
			vertex(
					map(i, 0, nsamples, 0, width),
					map(waveform.data[i], -1, 1, 0, height)
					);
		}
		
		endShape();

		fft.analyze(spectrum);

		for(int i = 0; i < bands; i++){

			/* the result of the FFT is normalized */
			/* draw the line for frequency band i scaling it up by 40 to get more amplitude */
			line( i, height, i, height - spectrum[i]*height*40);
			fftFeatures[i] = spectrum[i];
		} 

		fill(255);
		textSize(30);
		if(classifier != null) {
			String guessedLabel = classifier.classify(captureInstance(null));
			text("classified as: " + guessedLabel, 20, 30);
		}else {
			text(classNames[classIndex], 20, 30);
			dataCount = trainingData.get(classNames[classIndex]).size();
			text("Data collected: " + dataCount, 20, 60);
		}
	}
	
	public void keyPressed() {
		if (key == '.') {
			classIndex = (classIndex + 1) % classNames.length;
		}
		
		else if (key == 't') {
			if(classifier == null) {
				println("Start training ...");
				classifier = new MLClassifier();
				classifier.train(trainingData);
			}else {
				classifier = null;
			}
		}
		
		else if (key == 's') {
			// Yang: add code to save your trained model for later use
			try {
				Scanner sc = new Scanner(System.in); 
				System.out.print("Model name: ");  
				String name = sc.nextLine();  
				SerializationHelper.write(cwd + "/models/" + name + ".model", classifier);
//				File path = new File(cwd);
//				saveModel(classifier, classNames[classIndex], cwd);
				System.out.println("Trained model " + name + ".model SAVED!\n");
	        } catch(Exception e) {
	            e.printStackTrace();
	            System.out.println("Saving trained model FAILED!\n");
	        }
		}
		 
		else if (key == 'l') {
			// Yang: add code to load your previously trained model
			Scanner sc = new Scanner(System.in); 
			System.out.print("Load model: ");  
			String name = sc.nextLine(); 
			
			try {    
				classifier = (MLClassifier) weka.core.SerializationHelper.read(cwd + "/models/" + name + ".model");
//				File path = new File(cwd);
//				loadModel(path, classNames[classIndex]);
				System.out.println("Trained model " + name + ".model LOADED!\n");
	        } catch(Exception e) {
	            e.printStackTrace();
	            System.out.println("Load trained model " + name + ".model FAILED!\n");
	        }
		}
			
		else {
			trainingData.get(classNames[classIndex]).add(captureInstance(classNames[classIndex]));
		}
	}
	
//	private static void saveModel(MLClassifier c, String name, String path) throws Exception {
//	    ObjectOutputStream oos = null;
//	    System.out.println(path);
//	    try {
//	    	System.out.println("About to create new file to save..");
//	    	FileOutputStream new_file = new FileOutputStream(path + "/weka_models/" + name + ".model");
//	        oos = new ObjectOutputStream(new_file);
//	    } catch (FileNotFoundException e1) {
//	    	System.out.println("SAVE ERROR file not found");
//	        e1.printStackTrace();
//	    } catch (IOException e1) {
//	    	System.out.println("SAVE ERROR IOException");
//	        e1.printStackTrace();
//	    }
//	    System.out.println("did it fail????");
//	    oos.writeObject(c);
//	    oos.flush();
//	    oos.close();
//	}
//	
//	private static MLClassifier loadModel(File path, String name) throws Exception {
//
//		MLClassifier classifier;
//
//	    FileInputStream fis = new FileInputStream(path + name + ".model");
//	    ObjectInputStream ois = new ObjectInputStream(fis);
//
//	    classifier = (MLClassifier) ois.readObject();
//	    ois.close();
//
//	    return classifier;
//	}

}
