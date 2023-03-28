def sns_plot(feat, sns_function):
    fig, axis = plt.subplots(1,2)
    fig.suptitle(f"{feat} in test and train")

    sns_function(.)

fig, axis = plt.subplots(1,2)
fig.suptitle("Missing values in test and train")

sns.barplot(
    x = missing_train.index,
    y = missing_train.values,
    ax=axis[0],     # The left side
    palette="bright"
)
axis[0].set_title("Train")
axis[0].tick_params(axis = 'x', rotation=90)
axis[0].bar_label(axis[0].containers[0])

sns.barplot(
    x = missing_test.index,
    y = missing_test.values,
    ax=axis[1],     # The right side
    palette="dark"
)
axis[1].set_title("Test")
axis[1].tick_params(axis = 'x', rotation=90)
axis[1].bar_label(axis[1].containers[0])

plt.show()