from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
load_dotenv()

model1 = GoogleGenerativeAI(model = 'gemini-1.5-pro')

model2 = ChatGroq( model= "llama-3.1-8b-instant" )

prompt1 = PromptTemplate(
    template = "Generate short and simple note from the following text \n {text}",
    input_variables = ["text"]
)

prompt2 = PromptTemplate(
    template = " Generate short 5 questions and answers quiz from the given text \n {text}",
    input_variables = ["text"]
)
prompt3 = PromptTemplate(
    template = " merge the note and quiz into one document \n --> notes = {notes} \n --> quiz = {quiz}",
    input_variables = ["notes", "quiz"]
)

parser = StrOutputParser()



parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser 

chain = parallel_chain | merge_chain

text = """
Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. Note that the word “experiment” is not intended to denote academic use only, because even in commercial settings machine learning usually starts out experimentally. Here is a flowchart of typical cross validation workflow in model training. The best parameters can be determined by grid search techniques.

Grid Search Workflow
In scikit-learn a random split into training and test sets can be quickly computed with the train_test_split helper function. Let’s load the iris data set to fit a linear support vector machine on it:

When evaluating different settings (“hyperparameters”) for estimators, such as the C setting that must be manually set for an SVM, there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally. This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called “validation set”: training proceeds on the training set, after which evaluation is done on the validation set, and when the experiment seems to be successful, final evaluation can be done on the test set.

However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets.

A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow the same principles). The following procedure is followed for each of the k “folds”:

A model is trained using 
 of the folds as training data;

the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).

The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. This approach can be computationally expensive, but does not waste too much data (as is the case when fixing an arbitrary validation set), which is a major advantage in problems such as inverse inference where the number of samples is very small.

A depiction of a 5 fold cross validation on a training set, while holding out a test set.
3.1.1. Computing cross-validated metrics
The simplest way to use cross-validation is to call the cross_val_score helper function on the estimator and the dataset.


When the cv argument is an integer, cross_val_score uses the KFold or StratifiedKFold strategies by default, the latter being used if the estimator derives from ClassifierMixin.

It is also possible to use other cross validation strategies by passing a cross validation iterator instead, for instance:
"""

# try:
#     parallel_result = parallel_chain.invoke({'text': text})
#     print("Parallel chain result:", parallel_result)
#     final_result = merge_chain.invoke(parallel_result)
#     print("Final result:", final_result)
# except Exception as e:
#     print(f"Error occurred: {e}")

result = chain.invoke({'text' : text})
print(result)