package com.uttam;


import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.ClassAssigner;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class SpamClassifier {

	private static FilteredClassifier classifier = null;
	private static MultiFilter multiFilter = null;
	private final FastVector classVector;
	private final Attribute classAttribute;
	private final Attribute uniGrams;
	private final FastVector attributesVector;

	public SpamClassifier() {
		multiFilter = new MultiFilter();
		classVector = new FastVector(2);
		classVector.addElement("spam");
		classVector.addElement("nospam");
		classAttribute = new Attribute("the class", classVector);
		uniGrams = new Attribute("unigrams", (FastVector) null);
		attributesVector = new FastVector(2);
		attributesVector.addElement(classAttribute);
		attributesVector.addElement(uniGrams);
	}

	public Instances createTrainingInstances(){
		Instances trainingInstances = new Instances("Rel", attributesVector, 1);
		trainingInstances.setClassIndex(0);
		Instance instance = new Instance(2);
		instance.setValue(classAttribute, "spam");
		instance.setValue((Attribute) attributesVector.elementAt(1), "welcome to hdfc");
		trainingInstances.add(instance);
		trainingInstances.add(instance);
		trainingInstances.add(instance);
		trainingInstances.add(instance);
		instance = new Instance(2);
		instance.setValue(classAttribute, "nospam");
		instance.setValue((Attribute) attributesVector.elementAt(1), "hi dude");
		trainingInstances.add(instance);
		trainingInstances.add(instance);
		trainingInstances.add(instance);
		trainingInstances.add(instance);
		return trainingInstances;

	}

	public FilteredClassifier buildClassifier() throws Exception{
		Instances instances = createTrainingInstances();
		StringToWordVector filter = new StringToWordVector();
	    ClassAssigner classAssigner = new ClassAssigner();
	    classAssigner.setClassIndex("first");
	    multiFilter.setFilters(new Filter[] {filter, classAssigner});
	    multiFilter.setInputFormat(instances);
	    SMO svm = new SMO();
	    PolyKernel pk = new PolyKernel();
	    pk.setExponent(1);
	    svm.setKernel(pk);
	    classifier = new FilteredClassifier();
	    classifier.setFilter(multiFilter);
	    classifier.setClassifier(svm);

	    classifier.buildClassifier(instances);

	    Evaluation eval = new Evaluation(instances);
	    eval.evaluateModel(classifier, instances);
	    System.out.println(eval.toSummaryString("\nResults\n======\n", false));
	    System.out.println(classifier);
	    return classifier;
	}

	public String responseClass(String message) throws Exception{
		 Instances testingInstances = new Instances("Rel", attributesVector, 1);
		 testingInstances.setClassIndex(0);
		 Instance instance = new Instance(2);
		 instance.setValue((Attribute) attributesVector.elementAt(1),  message);
		 testingInstances.add(instance);
		 double pred = 0;
		 //this.wait(1000);
		 pred = classifier.classifyInstance(testingInstances.instance(0));
		 return testingInstances.classAttribute().value((int) pred);
	}


	public static void  main(String args[]) throws Exception {
		SpamClassifier spamClassifier = new SpamClassifier();
		spamClassifier.buildClassifier();
		System.out.println(spamClassifier.responseClass("welcome"));
	}

}
