import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def calculate_metrics(confusion_matrix, metrics="all"):
    
    TP = confusion_matrix[1][1]
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    
    #accuracy 
    acc = ( float (TP + TN) / float (TP + TN + FP + FN))
    
    #sensitivity
    sensitivity = ( TP / float ( TP + FN))
    
    #specificity
    specificity = ( TN / float (TN + FP))
    
    #Positive Predicted Value
    ppv = ( TP /  float ( TP + FP))
    
    #Negative Predicted Value
    npv = (TN / float ( TN + FN))  
    
    print(f"Accuracy:{acc:.4f}")
    print(f"Sensitivity:{sensitivity:.4f}")
    print(f"Specificity:{specificity:.4f}")
    print(f"Positive Predicted Value:{ppv:.4f}")
    print(f"Negative Predicted Value:{npv:.4f}")
      
data = {
    'Sample ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Gold standard': ['yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes'],
    'Test Method A': ['+', '-', '-', '+', '+', '-', '+', '+', '-', '+'],
    'Test Method B': ['+', '-', '-', '-', '-', '-', '+', '-', '+', '+']
}
df = pd.DataFrame(data)
df['Gold standard'] = df['Gold standard'].map({'yes':1,'no':0})
df['Test Method A'] = df['Test Method A'].map({'+':1, '-':0})
df['Test Method B'] = df['Test Method B'].map({'+':1, '-':0})



cm_a  = confusion_matrix(df['Gold standard'],df['Test Method A'])
cm_b = confusion_matrix(df['Gold standard'], df['Test Method B'])
#ConfusionMatrixDisplay(cm_a).plot()
#ConfusionMatrixDisplay(cm_b).plot()

print("\n----------The metrics for Test Method A are--------")
calculate_metrics(cm_a)
print("\n----------The metrics for Test Method B are--------")
calculate_metrics(cm_b)
