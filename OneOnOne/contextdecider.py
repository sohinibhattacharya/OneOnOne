class ContextDecider:
    def __int__(self, dataset, model, threshold):
        self.dataset = dataset
        self.model = model
        self.threshold = threshold

    def context_decider_for_bert(self, val_it, prediction_list):
        pred = self.model.predict_generator(val_it, 1)

        predicted_list = []
        classes_prob_list = []
        prediction_index = []
        final_classes = []

        output = []

        labels = self.val_it.class_indices
        labels2 = dict((v, k) for k, v in labels.items())

        for i in range(0, pred.shape[0]):
            classes_prob_list = []
            for j in range(0, self.output_layer_classes):
                classes_prob_list.append([int(j), pred[i][j]])

            classes_prob_list.sort(key=lambda x: x[1], reverse=True)

            first_highest_predicted_classes = classes_prob_list[0][0]
            first_highest_predicted_class_confidence = classes_prob_list[0][1]

            second_highest_predicted_classes = classes_prob_list[1][0]
            second_highest_predicted_class_confidence = classes_prob_list[1][1]

            predicted_list.append([first_highest_predicted_classes, first_highest_predicted_class_confidence,
                                   second_highest_predicted_classes, second_highest_predicted_class_confidence])

        if (predicted_list[0][1] - predicted_list[0][3]) >= 0.2:
            prediction_index.append(predicted_list[0][0])
        elif (predicted_list[0][1] - predicted_list[0][3]) < 0.2:
            prediction_index.append(predicted_list[0][0])
            prediction_index.append(predicted_list[0][2])

        for i in range(0, len(prediction_index)):
            final_classes.append(i2d[labels2[prediction_index[i]]])

        for ele in final_classes:
            b = ele.split(',')
            output = output + b

        return output

