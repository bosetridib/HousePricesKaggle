import matplotlib.pyplot as plt
import seaborn as sns

def sns_plot(feat, sns_function):
    
    fig, axis = plt.subplots(1,2)
    fig.suptitle(f"{feat} in test and train")
    
    sns_function = getattr(sns, sns_function)

    sns_function(
    x = feat.index,
    y = feat.values,
    ax=axis[0],     # The left side
    palette="bright"
    )
    axis[0].set_title("Train")

    sns_function(
    x = feat.index,
    y = feat.values,
    ax=axis[1],     # The right side
    palette="bright"
    )
    axis[1].set_title("Test")
    
    if 'barplot' in str(sns_function):
        axis[0].tick_params(axis = 'x', rotation=90)
        axis[0].bar_label(axis[0].containers[0])
        axis[1].tick_params(axis = 'x', rotation=90)
        axis[1].bar_label(axis[1].containers[0])
    
    plt.show()