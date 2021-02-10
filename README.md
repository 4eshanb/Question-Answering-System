# Question Answering System
This is a question answering system using stories from Aesop’s Fables, ROC stories corpus and MC test. 
The question answering (Q/A) system can process a story and a list
of questions, and produce an answer for each question. 

## NLP Pipeline
<img src="Diagrams/NLP-Pipeline.png" width="800">

The diagram as a whole refers to the NLP pipeline, but the blue specifically is the Question Answering Pipeline.


## Data 

Each story is segmented into sentences, and each story and sentence is assigned an id.
Each question comes from one of the stories, and has each of the questions have their own ids.
Each question id is also associated with a story id.
The answers are also provided, but purely for scoring purposes.  

The job of the QA system is to find the sentence id where the answer is located and then find the answer
text within this sentence. Sometimes there is more than one correct answer and sometimes there are no correct answers.  

## Scoring
Measures of performance are precision, recall, and f-measure.   
F-measure is the tradeoff between precision and recall:  
 f-measure = (2* precision * recall)/(recall + precision).  

Precision is the fraction of an answer that is considered correct. 
Recall is the fraction of an answer that is returned vs the complete answer text.  

<img src="Diagrams/Precision-Recall.png">

tp = true positive, which is an outcome where the model correctly predicts the positive class.   
tn = true negative, which is an outcome where the model correctly predicts the negative class.   
fp = false positive, which is an outcome where the model incorrectly predicts the positive class.  
fn = false negative, which is an outcome where the model incorrectly predicts the negative class.  

Precision = tp/(tp + fp)
In other words the precision is the percent of true positive over the number of positives.    
Recall = tp/(tp + fn)
The recall is the percent of true positives over the number of answers.  

For example, suppose the Question Answering system comes up with an answer to a given question, where the answer was:  
"Messi is phenomenal"  
However, the correct Answer is:  
"Lionel Messi"  
  
In this case, the recall would be 1/2 because the QA system found "Messi", but not "Lionel". 
The precision score is 1/3 because 1 out of the 3 words is correctly generated.  
The F-measure is:  
(2 * 1/2 * 1/3)/(1/2 + 1/3) = 1/3/(5/6) = 1/3 * 6/5 = 2/5 
                                                    = .4  

One can achieve a high recall if all the right answers are generated along with a lot of extra noise.  
On the other hand, high precision is when all returned answers must be correct.  
It is difficult to do well on both.  

>   Results we achieved:  
    AVERAGE RECALL =     0.6006  
    AVERAGE PRECISION = 0.5187   
    AVERAGE F-MEASURE = 0.4994  








