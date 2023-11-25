// Save the model (after training and testing) for loading and deployment
// https://waikato.github.io/weka-wiki/serialization/

import weka.core.SerializationHelper;

public Classifier load_model(Classifier cls, String filename){
    Classifier cls = (Classifier) weka.core.SerializationHelper.read(filename);
    return cls;
}