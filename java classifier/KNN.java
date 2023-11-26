// For Weka to find your class using its automatic Java class discovery mechanism when you want to run it in the GUIs 
//or from the command-line using weka.Run, it needs to be in the Java CLASSPATH and in one of Weka's standard Java packages 
//for classifiers (e.g., weka.classifiers.functions or weka.classifiers.trees). If that is the case, regardless of where 
//your class is physically located on your file system, it will show up in Weka’s GUIs automatically 
//(e.g., if you invoke the main() method of weka.gui.GUIChooser, which is the main entry point into Weka’s GUIs) 
//and also be available through weka.Run at a command-line interface.

// Note that the standard Java package structuring rules apply: 
//the directory structure for your class needs to match up with the fully qualified Java class name, 
//e.g., weka.classifiers.functions.MyFunctionalClassifier must be located in a folder called functions, 
//which in turn is located inside a folder called classifiers, which in turn is located in a folder called weka. 
//The folder containing this weka folder will need to be included in your CLASSPATH.

// On my computer, running macOS, having expanded weka-3-8-3.zip from the Weka website into /Users/eibe/weka-3-8-3, 
//and with the KNNMinimal.java file containing the above program in the folder /Users/eibe/weka-example/weka/classifiers/lazy, 
//I can use the following incantations to compile and run the classifier from the macOS command-line interface 
//(assuming the Java JDK has been installed):

// cd /Users/eibe/weka-example
// export CLASSPATH=/Users/eibe/weka-example:/Users/eibe/weka-3-8-3/weka.jar
// javac weka/classifiers/lazy/KNNMinimal.java
// java weka.Run .KNNMinimal -t /Users/eibe/weka-3-8-3/data/iris.arff
// This will run a 10-fold cross-validation with our 1-nearest-neighbour classifier on the iris data. 
//And, to start up the Weka GUIs and use the classifier from those, we can enter

// java weka.gui.GUIChooser
// Section 2: Options, capabilities, and textual output
// For Weka’s GUIs to work properly with your Classifier class, it needs to implement Java's Serializable indicator interface. 
//AbstractClassifier does that, so the above example code will work fine. AbstractClassifier 
//also implements a bunch of other interfaces, including the OptionHandler interface that is used for 
//command-line option handling. There are four command-line options already implemented in AbstractClassifier, 
//which are automatically added to the -K option we have specified in the above example classifier:

// -output-debug-info
// -do-not-check-capabiliities
// -num-decimal-places <int>
// -batch-size <int>
// The first option will simply set the protected member variable m_Debug to true. 
//You can use it in your class to output optional debug information, or you can just ignore it. 
//The second option is only relevant if your class implements handling of capabilities. 
//More on that in a second. The third option sets the value of the m_numDecimalPlaces variable. 
//This should be used in the toString() method of your class, which you need to implement if you want a textual 
//description of your model to be output by Weka, to specify the number of significant digits 
//that are used when floating-point numbers are included in the output. 
//The fourth option is ignored by almost all classifiers in Weka: 
//it can be used to set a desired batch size for batch prediction when the classifier is used in batch prediction mode.

// Below is an expanded version of the above example code that includes a toString() method and a getCapabilities() method. 
//The toString() method in this example code is rudimentary and just outputs the number of neighbours used by the classifier. 
//The biggest method is the getCapabilities() method. This method is optional. 
//It specifies what kind of data this classifier is able to deal with and is used in Weka’s GUIs to grey out a classifier 
//if it is not applicable to a particular dataset. It is also used in the buildClassifier(Instances) method 
//in this example code: getCapabilities().testWithFail(trainingData) will use it to check whether the classifier 
//is actually applicable to the data provided for training. Note that implementing this method is really optional: 
//AbstractClassifier has a default implementation of getCapabilities() that does not restrict the classifier in any way. 
//Basically, getCapabilities() only needs to be implemented if you want your classifier to be used by other users, 
//to make it more user friendly.


/**
* This code is released to the public domain. Use as you see fit.
*/
package weka.classifiers.lazy;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.core.Capabilities;
import weka.classifiers.AbstractClassifier;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
* Implements the k-nearest-neighbours method for classification and
* regression.  Existing WEKA code is used to retrieve the K nearest
* neighbours for a test instance. The number of neighbours to use is
* a parameter that the user can specify, via a get...()/set...()
* method pair for WEKA's GUIs and a Java annotation for command-line
* option handling.
*/

public class KNN extends AbstractClassifier {

   /** The number of neighbours to use */
   protected int m_K = 1;

   /** The method to be used to search for nearest neighbours. */
   protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

   /**
    * Returns capabilities of the classifier.
    *
    * @return the capabilities of this classifier
    */
   public Capabilities getCapabilities() {
       Capabilities result = super.getCapabilities();
       result.disableAll();

       // predictor attributes
       result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
       result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
       result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
       result.enable(Capabilities.Capability.MISSING_VALUES);

       // class
       result.enable(Capabilities.Capability.NOMINAL_CLASS);
       result.enable(Capabilities.Capability.NUMERIC_CLASS);
       result.enable(Capabilities.Capability.DATE_CLASS);
       result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

       return result;
   }

   /**
    * Method to set the number of neighbours. Including metadata annotation
    * to implement command-line option handling for this parameter.
    */
   @OptionMetadata(displayName = "number of neighbours", description = "Number of neighbours to use (default = 1).",
                   commandLineParamName = "K", commandLineParamSynopsis = "-K <int>", displayOrder = 1)
   public void setK(int k) {
       m_K = k;
   }

   /**
    * Method to get the currently set number of neighbours.
    */
   public int getK() {
       return m_K;
   }

   /**
    * Initialises the classifier from the given training instances.
    */
   public void buildClassifier(Instances trainingData) throws Exception {

       // Can the classifier handle the data?
       getCapabilities().testWithFail(trainingData);

       // Make a copy of data and delete instances with a missing class value
       trainingData = new Instances(trainingData);
       trainingData.deleteWithMissingClass();

       // Trivial for KNN: just initialise NN search class
       m_NNSearch.setInstances(trainingData);
   }

   /**
    * Returns class probability distribution (classification) or numeric
    * target value (regression) for a given test instance.
    */
   public double[] distributionForInstance(Instance testInstance) throws Exception {

       // Add instance to NN search so that attribute ranges can be updated
       m_NNSearch.addInstanceInfo(testInstance);

       // Get the list of neighbours
       Instances neighbours = m_NNSearch.kNearestNeighbours(testInstance, m_K);

       // Calculate calculate class probability distribution or target value
       double[] dist = new double[testInstance.numClasses()];
       for (Instance neighbour : neighbours) {
           if (testInstance.classAttribute().isNominal()) {
               dist[(int)neighbour.classValue()] += 1.0 / neighbours.numInstances();
           } else {
               dist[0] += neighbour.classValue() / neighbours.numInstances();
           }
       }
       return dist;
   }

   /**
    * Returns a textual description of the classifier.
    */
   public String toString() {

       // Not much to output here for KNN: no explicit model
       return "KNN with " + m_K + " neighbours";
   }
}