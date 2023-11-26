// Modified from https://waikato.github.io/weka-wiki/serialization/
// Load the model for use
import weka.core.SerializationHelper;

public Classifier load_model(Classifier cls, String filename){
    Classifier cls = (Classifier) weka.core.SerializationHelper.read(filename);
    return cls;
}