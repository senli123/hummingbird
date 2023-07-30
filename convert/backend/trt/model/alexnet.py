from .base_model import BaseModel
import tensorrt as trt
class alexnet(BaseModel):
    def create_engine(self,
                      builder,
                      config, 
                      wts_path,
                      dt,
                      input_blob_name,
                    output_blob_name,
                    max_batch_size, 
                    input_h,
                    input_w
                            ):
        weight_map = self.load_weights(wts_path)
        network = builder.create_network()

        data = network.add_input(input_blob_name, dt, (3, input_h, input_w))
        assert data

        conv1 = network.add_convolution(input=data,
                                        num_output_maps=64,
                                        kernel_shape=(11, 11),
                                        kernel=weight_map["features.0.weight"],
                                        bias=weight_map["features.0.bias"])
        assert conv1
        conv1.stride = (4, 4)
        conv1.padding = (2, 2)

        relu1 = network.add_activation(conv1.get_output(0), type=trt.ActivationType.RELU)
        assert relu1

        pool1 = network.add_pooling(input=relu1.get_output(0),
                                    type=trt.PoolingType.MAX,
                                    window_size=trt.DimsHW(3, 3))
        assert pool1
        pool1.stride_nd = (2, 2)

        conv2 = network.add_convolution(input=pool1.get_output(0),
                                        num_output_maps=192,
                                        kernel_shape=(5, 5),
                                        kernel=weight_map["features.3.weight"],
                                        bias=weight_map["features.3.bias"])
        assert conv2
        conv2.padding = (2, 2)

        relu2 = network.add_activation(conv2.get_output(0), type=trt.ActivationType.RELU)
        assert relu2

        pool2 = network.add_pooling(input=relu2.get_output(0),
                                    type=trt.PoolingType.MAX,
                                    window_size=trt.DimsHW(3, 3))
        assert pool2
        pool2.stride_nd = (2, 2)

        conv3 = network.add_convolution(input=pool2.get_output(0),
                                        num_output_maps=384,
                                        kernel_shape=(3, 3),
                                        kernel=weight_map["features.6.weight"],
                                        bias=weight_map["features.6.bias"])
        assert conv3
        conv3.padding = (1, 1)

        relu3 = network.add_activation(conv3.get_output(0), type=trt.ActivationType.RELU)
        assert relu3

        conv4 = network.add_convolution(input=relu3.get_output(0),
                                        num_output_maps=256,
                                        kernel_shape=(3, 3),
                                        kernel=weight_map["features.8.weight"],
                                        bias=weight_map["features.8.bias"])
        assert conv4
        conv4.padding = (1, 1)

        relu4 = network.add_activation(conv4.get_output(0), type=trt.ActivationType.RELU)
        assert relu4

        conv5 = network.add_convolution(input=relu4.get_output(0),
                                        num_output_maps=256,
                                        kernel_shape=(3, 3),
                                        kernel=weight_map["features.10.weight"],
                                        bias=weight_map["features.10.bias"])
        assert conv5
        conv5.padding = (1, 1)

        relu5 = network.add_activation(conv5.get_output(0), type=trt.ActivationType.RELU)
        assert relu5

        pool3 = network.add_pooling(input=relu5.get_output(0),
                                    type=trt.PoolingType.MAX,
                                    window_size=trt.DimsHW(3, 3))
        assert pool3
        pool3.stride_nd = (2, 2)

        fc1 = network.add_fully_connected(input=pool3.get_output(0),
                                        num_outputs=4096,
                                        kernel=weight_map["classifier.1.weight"],
                                        bias=weight_map["classifier.1.bias"])
        assert fc1

        relu6 = network.add_activation(fc1.get_output(0), type=trt.ActivationType.RELU)
        assert relu6

        fc2 = network.add_fully_connected(input=relu6.get_output(0),
                                        num_outputs=4096,
                                        kernel=weight_map["classifier.4.weight"],
                                        bias=weight_map["classifier.4.bias"])
        assert fc2

        relu7 = network.add_activation(fc2.get_output(0), type=trt.ActivationType.RELU)
        assert relu7

        fc3 = network.add_fully_connected(input=relu7.get_output(0),
                                        num_outputs=1000,
                                        kernel=weight_map["classifier.6.weight"],
                                        bias=weight_map["classifier.6.bias"])
        assert fc3

        fc3.get_output(0).name = output_blob_name
        network.mark_output(fc3.get_output(0))

        # Build Engine
        builder.max_batch_size = max_batch_size
        builder.max_workspace_size = 1 << 20
        engine = builder.build_engine(network, config)

        del network
        del weight_map

        return engine
            