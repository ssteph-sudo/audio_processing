// Modified from https://waikato.github.io/weka-wiki/serialization/
// Save the model (after training and testing) for loading and deployment
import weka.core.SerializationHelper;

public void save_model(Classifier cls, String filename){
    weka.core.SerializationHelper.write(filename, cls);
}