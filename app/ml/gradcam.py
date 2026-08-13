import torch
import torch.nn.functional as F


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, target_class=None):

        self.model.zero_grad()

        output = self.model(image_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        target_score = output[:, target_class]

        target_score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (weights * activations).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return (
            cam.detach(),
            output.detach(),
            target_class,
        )

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()