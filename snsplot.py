import matplotlib.pyplot as plt
import seaborn as sns

def sns_plot(feat_train, feat_test, sns_function):
    
    fig, axis = plt.subplots(1,2)
    fig.suptitle("Column in test and train")

    if 'barplot' in str(sns_function):
        sns_function = getattr(sns, sns_function)
        sns_function(
            x = feat_train.index,
            y = feat_train.values,
            ax=axis[0],     # The left side
            palette="bright"
        )
        axis[0].set_title("Train")
        axis[0].tick_params(axis = 'x', rotation=90)
        axis[0].bar_label(axis[0].containers[0])

        
        sns_function(
            x = feat_test.index,
            y = feat_test.values,
            ax=axis[1],     # The right side
            palette="bright"
        )
        axis[1].set_title("Test")
        axis[1].tick_params(axis = 'x', rotation=90)
        axis[1].bar_label(axis[1].containers[0])
    
    elif 'histplot' in str(sns_function):
        sns_function = getattr(sns, sns_function)
        sns_function(
            feat_train,
            ax=axis[0],     # The left side
        )
        axis[0].set_title("Train")
        
        sns_function(
            feat_test,
            ax=axis[1],     # The right side
        )
        axis[1].set_title("Test")
    
    else:
        print("The plot doesn't exist in snsplot.py")
    
    plt.show()