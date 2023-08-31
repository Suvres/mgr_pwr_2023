custom_labels <- function(dataframe1) {
 #
# Change the "label.type" attribute of column "Class" to be of type "True Labels"
    attr(data.set$'Label', "label.type") <- "True Labels"
#
# Change the "feature.channel" and "score.type" attribute of "scored" column
    attr(data.set$'Scored Labels', "feature.channel") <- "Binary Classification Scores"
    attr(data.set$'Scored Labels', "score.type") <- "Assigned Labels"
#
# Change the "feature.channel" and "score.type" attribute of "probs" column
    attr(data.set$'Scored Probabilities', "feature.channel") <- "Binary Classification Scores"
    attr(data.set$'Scored Probabilities', "score.type") <- "Calibrated Score"

}