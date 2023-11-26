// Modified from the source below:
// https://waikato.github.io/weka-blog/posts/2018-10-08-making-a-weka-classifier/

// classifyInstance(Instance), which takes a test instance and returns a single predicted class value for the instance 
// that is supplied, and 
// distributionForInstance(Instance), which returns a class probability distribution instead, 
// assuming the class attribute is nominal. The fourth method in the Classifier interface, 
// getCapabilities(), returns the capabilities of the classifier, 
// specifying what kind of data it can be applied to. More on that further below.


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


// Weka’s AbstractClassifier class is the superclass of all Classifier classes found in Weka. 
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


   //Build a model based on certain training instances
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


// One more thing: if you want your class to be located in a new Java package that is not one of Weka’s 
// standard packages for classifiers, you will need to make an appropriate version of the 
// GenericPropertiesCreator.props file for Weka. For example, the RPlugin package for Weka 
// defines a new weka.classifiers.mlr package and has the following info in the GenericPropertiesCreator.props file:

// weka.classifiers.Classifier=\
// weka.classifiers.mlr
// That is it from me for today. Hope you are finding this useful.


// https://medium.com/mlearning-ai/exploring-the-java-weka-machine-learning-library-48e842b88307