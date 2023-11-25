// Modified from https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/#traintest-set
import weka.classifiers.Evaluation;

public void print_metrics(Instances train, Instances test, Classifier cls){

    // evaluate classifier and print some statistics
    Evaluation eval = new Evaluation(train);
    eval.evaluateModel(cls, test);
    System.out.println(eval.toSummaryString("\nResults\n======\n", false));

}
