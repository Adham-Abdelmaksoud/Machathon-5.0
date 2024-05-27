def train_cnn(
        model,
        images_train,
        labels_train,
        epochs,
        lossFunction,
        optimizer,
        print_loss=False,
):

    for i in range(epochs):
        losses_batch = []
        for b in range(len(images_train)):
            y_pred = model.forward(images_train[b])

            loss_train = lossFunction(y_pred, labels_train[b])
            losses_batch.append(loss_train.detach().numpy())

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        if print_loss:
            print(f'Epoch: {i} / loss: {loss_train}')

    return model