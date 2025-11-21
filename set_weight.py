def set_weight_PhaseNet(model, model_weights):

    # 重みの対応付けを行い、Kerasモデルに設定する
    # initial_conv
    model.get_layer(name='initial_conv').set_weights([
        model_weights['inc.weight'].numpy().transpose(2, 1, 0),
        model_weights['inc.bias'].numpy()
    ])
    
    # initial_bn
    model.get_layer(name='initial_bn').set_weights([
        model_weights['in_bn.weight'].numpy(),
        model_weights['in_bn.bias'].numpy(),
        model_weights['in_bn.running_mean'].numpy(),
        model_weights['in_bn.running_var'].numpy()
    ])
    
    # down1_conv1
    model.get_layer(name='down1_conv1').set_weights([
        model_weights['down_branch.0.0.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # down1_bn1
    model.get_layer(name='down1_bn1').set_weights([
        model_weights['down_branch.0.1.weight'].numpy(),
        model_weights['down_branch.0.1.bias'].numpy(),
        model_weights['down_branch.0.1.running_mean'].numpy(),
        model_weights['down_branch.0.1.running_var'].numpy()
    ])
    
    # down1_conv2
    model.get_layer(name='down1_conv2').set_weights([
        model_weights['down_branch.0.2.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # down1_bn2
    model.get_layer(name='down1_bn2').set_weights([
        model_weights['down_branch.0.3.weight'].numpy(),
        model_weights['down_branch.0.3.bias'].numpy(),
        model_weights['down_branch.0.3.running_mean'].numpy(),
        model_weights['down_branch.0.3.running_var'].numpy()
    ])
    
    # down2_conv1
    model.get_layer(name='down2_conv1').set_weights([
        model_weights['down_branch.1.0.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # down2_bn1
    model.get_layer(name='down2_bn1').set_weights([
        model_weights['down_branch.1.1.weight'].numpy(),
        model_weights['down_branch.1.1.bias'].numpy(),
        model_weights['down_branch.1.1.running_mean'].numpy(),
        model_weights['down_branch.1.1.running_var'].numpy()
    ])
    
    # down2_conv2
    model.get_layer(name='down2_conv2').set_weights([
        model_weights['down_branch.1.2.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # down2_bn2
    model.get_layer(name='down2_bn2').set_weights([
        model_weights['down_branch.1.3.weight'].numpy(),
        model_weights['down_branch.1.3.bias'].numpy(),
        model_weights['down_branch.1.3.running_mean'].numpy(),
        model_weights['down_branch.1.3.running_var'].numpy()
    ])
    
    # down3_conv1
    model.get_layer(name='down3_conv1').set_weights([
        model_weights['down_branch.2.0.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # down3_bn1
    model.get_layer(name='down3_bn1').set_weights([
        model_weights['down_branch.2.1.weight'].numpy(),
        model_weights['down_branch.2.1.bias'].numpy(),
        model_weights['down_branch.2.1.running_mean'].numpy(),
        model_weights['down_branch.2.1.running_var'].numpy()
    ])
    
    # down3_conv2
    model.get_layer(name='down3_conv2').set_weights([
        model_weights['down_branch.2.2.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # down3_bn2
    model.get_layer(name='down3_bn2').set_weights([
        model_weights['down_branch.2.3.weight'].numpy(),
        model_weights['down_branch.2.3.bias'].numpy(),
        model_weights['down_branch.2.3.running_mean'].numpy(),
        model_weights['down_branch.2.3.running_var'].numpy()
    ])
    
    # down4_conv1
    model.get_layer(name='down4_conv1').set_weights([
        model_weights['down_branch.3.0.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # down4_bn1
    model.get_layer(name='down4_bn1').set_weights([
        model_weights['down_branch.3.1.weight'].numpy(),
        model_weights['down_branch.3.1.bias'].numpy(),
        model_weights['down_branch.3.1.running_mean'].numpy(),
        model_weights['down_branch.3.1.running_var'].numpy()
    ])
    
    # down4_conv2
    model.get_layer(name='down4_conv2').set_weights([
        model_weights['down_branch.3.2.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # down4_bn2
    model.get_layer(name='down4_bn2').set_weights([
        model_weights['down_branch.3.3.weight'].numpy(),
        model_weights['down_branch.3.3.bias'].numpy(),
        model_weights['down_branch.3.3.running_mean'].numpy(),
        model_weights['down_branch.3.3.running_var'].numpy()
    ])
    
    # down5_conv1
    model.get_layer(name='down5_conv1').set_weights([
        model_weights['down_branch.4.0.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # down5_bn1
    model.get_layer(name='down5_bn1').set_weights([
        model_weights['down_branch.4.1.weight'].numpy(),
        model_weights['down_branch.4.1.bias'].numpy(),
        model_weights['down_branch.4.1.running_mean'].numpy(),
        model_weights['down_branch.4.1.running_var'].numpy()
    ])
    
    # up4_conv1
    model.get_layer(name='up4_conv1').set_weights([
        model_weights['up_branch.0.0.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # up4_bn1
    model.get_layer(name='up4_bn1').set_weights([
        model_weights['up_branch.0.1.weight'].numpy(),
        model_weights['up_branch.0.1.bias'].numpy(),
        model_weights['up_branch.0.1.running_mean'].numpy(),
        model_weights['up_branch.0.1.running_var'].numpy()
    ])
    
    # up4_conv2
    model.get_layer(name='up4_conv2').set_weights([
        model_weights['up_branch.0.2.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # up4_bn2
    model.get_layer(name='up4_bn2').set_weights([
        model_weights['up_branch.0.3.weight'].numpy(),
        model_weights['up_branch.0.3.bias'].numpy(),
        model_weights['up_branch.0.3.running_mean'].numpy(),
        model_weights['up_branch.0.3.running_var'].numpy()
    ])
    
    # up3_conv1
    model.get_layer(name='up3_conv1').set_weights([
        model_weights['up_branch.1.0.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # up3_bn1
    model.get_layer(name='up3_bn1').set_weights([
        model_weights['up_branch.1.1.weight'].numpy(),
        model_weights['up_branch.1.1.bias'].numpy(),
        model_weights['up_branch.1.1.running_mean'].numpy(),
        model_weights['up_branch.1.1.running_var'].numpy()
    ])
    
    # up3_conv2
    model.get_layer(name='up3_conv2').set_weights([
        model_weights['up_branch.1.2.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # up3_bn2
    model.get_layer(name='up3_bn2').set_weights([
        model_weights['up_branch.1.3.weight'].numpy(),
        model_weights['up_branch.1.3.bias'].numpy(),
        model_weights['up_branch.1.3.running_mean'].numpy(),
        model_weights['up_branch.1.3.running_var'].numpy()
    ])
    
    # up2_conv1
    model.get_layer(name='up2_conv1').set_weights([
        model_weights['up_branch.2.0.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # up2_bn1
    model.get_layer(name='up2_bn1').set_weights([
        model_weights['up_branch.2.1.weight'].numpy(),
        model_weights['up_branch.2.1.bias'].numpy(),
        model_weights['up_branch.2.1.running_mean'].numpy(),
        model_weights['up_branch.2.1.running_var'].numpy()
    ])
    
    # up2_conv2
    model.get_layer(name='up2_conv2').set_weights([
        model_weights['up_branch.2.2.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # up2_bn2
    model.get_layer(name='up2_bn2').set_weights([
        model_weights['up_branch.2.3.weight'].numpy(),
        model_weights['up_branch.2.3.bias'].numpy(),
        model_weights['up_branch.2.3.running_mean'].numpy(),
        model_weights['up_branch.2.3.running_var'].numpy()
    ])
    
    # up1_conv1
    model.get_layer(name='up1_conv1').set_weights([
        model_weights['up_branch.3.0.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # up1_bn1
    model.get_layer(name='up1_bn1').set_weights([
        model_weights['up_branch.3.1.weight'].numpy(),
        model_weights['up_branch.3.1.bias'].numpy(),
        model_weights['up_branch.3.1.running_mean'].numpy(),
        model_weights['up_branch.3.1.running_var'].numpy()
    ])
    
    # up1_conv2
    model.get_layer(name='up1_conv2').set_weights([
        model_weights['up_branch.3.2.weight'].numpy().transpose(2, 1, 0)
    ])
    
    # up1_bn2
    model.get_layer(name='up1_bn2').set_weights([
        model_weights['up_branch.3.3.weight'].numpy(),
        model_weights['up_branch.3.3.bias'].numpy(),
        model_weights['up_branch.3.3.running_mean'].numpy(),
        model_weights['up_branch.3.3.running_var'].numpy()
    ])
    
    # output_conv
    model.get_layer(name='output_conv').set_weights([
        model_weights['out.weight'].numpy().transpose(2, 1, 0),
        model_weights['out.bias'].numpy()
    ])

    return model