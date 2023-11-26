import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import "Get_Training_Data.java";
 

public static void main(String[] args) {

    String csv_file_path = "../feature_extraction/features.csv"
    
    Instances data = read_training_data(csv_file_path)

    // Classifier cls = new J48();
    // cls.buildClassifier(train);
}