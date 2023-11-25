// Just copied and pasted from the source below for now:
// https://waikato.github.io/weka-blog/posts/2018-10-08-making-a-weka-classifier/
// Will modify later

// WEKA (implementation of a collection of machine learning algorithms in 
// Java, downloadable at http://www.cs.waikato.ac.nz/ml/weka/

// https://www.cs.waikato.ac.nz/ml/weka/mooc/dataminingwithweka/ << instructions
// There are three primary methods in the Classifier interface: 
// buildClassifier(Instances), which will build the classification or regression model based on the given training instances, 
// classifyInstance(Instance), which takes a test instance and returns a single predicted class value for the instance 
// that is supplied, and 
// distributionForInstance(Instance), which returns a class probability distribution instead, 
// assuming the class attribute is nominal. The fourth method in the Classifier interface, 
// getCapabilities(), returns the capabilities of the classifier, 
// specifying what kind of data it can be applied to. More on that further below.

// In practice, it is normally best to just extend Weka’s AbstractClassifier class, 
// which implements the Classifier interface and is the superclass of pretty much all 
// Classifier classes that can be found in Weka. 
// It has default implementations for the
// distributionForInstance(Instance) and 
// classifyInstance(Instance) methods, one of which you have to override in your Classifier class. 
// You also need to implement the buildClassifier(Instances) method to actually build the classification/regression model. 
// The AbstractClassifier has a default implementation of the getCapabilities() method, 
// which simply returns all possible capabilities, implying that the classifier can be applied to any kind of training data. 
// Your class will inherit this default implementation and overriding this method is optional.

// So, at a minimum, the Java class for your learning algorithm, 
// assuming it extends AbstractClassifier, needs to implement two methods: 
// buildClassifier(Instances) and 
// distributionForInstance(Instance) 
// (or classifyInstance(Instance)). 
// The distributionForInstance(Instance) method must return a double array 
// containing the estimated class probabilities for the different class values of the test instance if the class is nominal. 
// If the class is numeric, it must return a single-element array with the numeric prediction for the test instance. 
// The classifyInstance(Instance) method, if you choose to implement it instead of 
// distributionForInstance(Instance), must return the index of the predicted class value (coded as a number of type double) 
// if the class is nominal and simply return the predicted class value if the class is numeric.

// Below you can find an implementation of a basic K-nearest-neighbours classifier for Weka. 
// It implements buildClassifier(Instances) and distributionForInstance(Instance), 
// the two required methods for a Classifier. 
// It also has a setter/getter method pair to set the value of the parameter K in WEKA's 
// graphical user interfaces (GUIs) and shows WEKA's @OptionMetadata annotation, 
// which is used to specify further information for this parameter, e.g., 
// the corresponding command-line option -K <int>. 
// This is all the code that is necessary to use the classifier in Weka’s 
// GUIs or run it from a command-line interface using the weka.Run class. 
// For example, it would be enough to run systematic experiments with the 
// K-nearest-neighbour method in Weka’s Experimenter GUI. 
// As you can see, it is pretty straightforward to implement a classifier in Weka!

package weka.classifiers.lazy;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.classifiers.AbstractClassifier;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class KNNMinimal extends AbstractClassifier {

   protected int m_K = 1;

   protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

   @OptionMetadata(displayName = "number of neighbours", description = "Number of neighbours to use (default = 1).",
                   commandLineParamName = "K", commandLineParamSynopsis = "-K <int>", displayOrder = 1)
   public void setK(int k) {
       m_K = k;
   }

   public int getK() {
       return m_K;
   }

   public void buildClassifier(Instances trainingData) throws Exception {

       trainingData = new Instances(trainingData);
       trainingData.deleteWithMissingClass();

       m_NNSearch.setInstances(trainingData);
   }

   public double[] distributionForInstance(Instance testInstance) throws Exception {

       m_NNSearch.addInstanceInfo(testInstance);

       Instances neighbours = m_NNSearch.kNearestNeighbours(testInstance, m_K);

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
}
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

// One more thing: if you want your class to be located in a new Java package that is not one of Weka’s 
// standard packages for classifiers, you will need to make an appropriate version of the 
// GenericPropertiesCreator.props file for Weka. For example, the RPlugin package for Weka 
// defines a new weka.classifiers.mlr package and has the following info in the GenericPropertiesCreator.props file:

// weka.classifiers.Classifier=\
// weka.classifiers.mlr
// That is it from me for today. Hope you are finding this useful.


// https://medium.com/mlearning-ai/exploring-the-java-weka-machine-learning-library-48e842b88307