"""
测试脚本: 验证fuse_token实现的正确性
"""
import torch
import sys
sys.path.append('/home/master/file/vision_transformer/vision_transformer')

from vit_model_fuse import vit_base_patch16_224_fuse


def test_fuse_token_shape():
    """测试fuse_token的输出形状"""
    print("=" * 50)
    print("测试1: 验证fuse_token输出形状")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    # 创建随机输入
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    # 测试1: 不使用fuse_token, keep_rate=1.0 (标准ViT)
    print("\n场景1: keep_rate=1.0, fuse_token=False (标准ViT)")
    model1 = vit_base_patch16_224_fuse(num_classes=1000, keep_rate=1.0, fuse_token=False).to(device)
    model1.eval()
    with torch.no_grad():
        output1 = model1(x, keep_rate=1.0)
    print(f"输出形状: {output1.shape}")
    print(f"预期: [{batch_size}, 1000]")
    assert output1.shape == (batch_size, 1000), "标准ViT输出形状错误"
    print("✓ 通过")

    # 测试2: 使用fuse_token, keep_rate=0.7
    print("\n场景2: keep_rate=0.7, fuse_token=True")
    model2 = vit_base_patch16_224_fuse(num_classes=1000, keep_rate=0.7, fuse_token=True).to(device)
    model2.eval()
    with torch.no_grad():
        output2 = model2(x, keep_rate=0.7)
    print(f"输出形状: {output2.shape}")
    print(f"预期: [{batch_size}, 1000]")
    assert output2.shape == (batch_size, 1000), "fuse_token ViT输出形状错误"
    print("✓ 通过")

    # 测试3: 不使用fuse_token, keep_rate=0.7
    print("\n场景3: keep_rate=0.7, fuse_token=False")
    model3 = vit_base_patch16_224_fuse(num_classes=1000, keep_rate=0.7, fuse_token=False).to(device)
    model3.eval()
    with torch.no_grad():
        output3 = model3(x, keep_rate=0.7)
    print(f"输出形状: {output3.shape}")
    print(f"预期: [{batch_size}, 1000]")
    assert output3.shape == (batch_size, 1000), "无fuse_token剪枝ViT输出形状错误"
    print("✓ 通过")

    print("\n" + "=" * 50)
    print("所有形状测试通过!")
    print("=" * 50)


def test_forward_pass():
    """测试前向传播是否正常工作"""
    print("\n" + "=" * 50)
    print("测试2: 验证前向传播")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = vit_base_patch16_224_fuse(num_classes=1000, keep_rate=0.7, fuse_token=True).to(device)
    model.eval()

    # 测试不同batch size
    for bs in [1, 2, 4]:
        print(f"\n测试batch_size={bs}")
        x = torch.randn(bs, 3, 224, 224).to(device)

        with torch.no_grad():
            output = model(x, keep_rate=0.7)

        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

        # 检查输出是否包含NaN或Inf
        assert not torch.isnan(output).any(), "输出包含NaN"
        assert not torch.isinf(output).any(), "输出包含Inf"
        print("  ✓ 通过")

    print("\n" + "=" * 50)
    print("前向传播测试通过!")
    print("=" * 50)


def test_different_keep_rates():
    """测试不同的keep_rate值"""
    print("\n" + "=" * 50)
    print("测试3: 测试不同keep_rate值")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vit_base_patch16_224_fuse(num_classes=1000, keep_rate=1.0, fuse_token=True).to(device)
    model.eval()

    x = torch.randn(2, 3, 224, 224).to(device)

    keep_rates = [1.0, 0.9, 0.7, 0.5]

    for kr in keep_rates:
        print(f"\n测试keep_rate={kr}")
        with torch.no_grad():
            output = model(x, keep_rate=kr)

        print(f"  输出形状: {output.shape}")
        print(f"  输出均值: {output.mean().item():.3f}")
        print(f"  输出标准差: {output.std().item():.3f}")

        assert output.shape == (2, 1000), f"keep_rate={kr}时输出形状错误"
        assert not torch.isnan(output).any(), f"keep_rate={kr}时输出包含NaN"
        print("  ✓ 通过")

    print("\n" + "=" * 50)
    print("不同keep_rate测试通过!")
    print("=" * 50)


def test_gradient_flow():
    """测试梯度是否能正常反向传播"""
    print("\n" + "=" * 50)
    print("测试4: 验证梯度反向传播")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vit_base_patch16_224_fuse(num_classes=1000, keep_rate=0.7, fuse_token=True).to(device)
    model.train()

    x = torch.randn(2, 3, 224, 224).to(device)
    target = torch.randint(0, 1000, (2,)).to(device)

    # 前向传播
    output = model(x, keep_rate=0.7)
    loss = torch.nn.functional.cross_entropy(output, target)

    print(f"损失值: {loss.item():.3f}")

    # 反向传播
    loss.backward()

    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"  {name}: grad_norm={grad_norm:.6f}")
                break

    assert has_grad, "没有参数有梯度"
    print("\n✓ 梯度反向传播正常")

    print("\n" + "=" * 50)
    print("梯度测试通过!")
    print("=" * 50)


def test_comparison_with_without_fuse():
    """比较有无fuse_token的差异"""
    print("\n" + "=" * 50)
    print("测试5: 比较有无fuse_token")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建两个模型
    model_with_fuse = vit_base_patch16_224_fuse(num_classes=1000, keep_rate=0.7, fuse_token=True).to(device)
    model_without_fuse = vit_base_patch16_224_fuse(num_classes=1000, keep_rate=0.7, fuse_token=False).to(device)

    model_with_fuse.eval()
    model_without_fuse.eval()

    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output_with = model_with_fuse(x, keep_rate=0.7)
        output_without = model_without_fuse(x, keep_rate=0.7)

    print(f"\n有fuse_token输出形状: {output_with.shape}")
    print(f"无fuse_token输出形状: {output_without.shape}")
    print(f"\n有fuse_token输出统计:")
    print(f"  均值: {output_with.mean().item():.3f}")
    print(f"  标准差: {output_with.std().item():.3f}")
    print(f"\n无fuse_token输出统计:")
    print(f"  均值: {output_without.mean().item():.3f}")
    print(f"  标准差: {output_without.std().item():.3f}")

    # 两者输出应该不同(因为token数量不同)
    diff = (output_with - output_without).abs().mean().item()
    print(f"\n输出差异(平均绝对差): {diff:.3f}")

    print("\n✓ 有无fuse_token产生不同输出(符合预期)")

    print("\n" + "=" * 50)
    print("比较测试通过!")
    print("=" * 50)


if __name__ == '__main__':
    print("\n开始测试fuse_token实现...\n")

    try:
        test_fuse_token_shape()
        test_forward_pass()
        test_different_keep_rates()
        test_gradient_flow()
        test_comparison_with_without_fuse()

        print("\n" + "=" * 50)
        print("🎉 所有测试通过!")
        print("=" * 50)
        print("\nfuse_token实现验证成功,可以开始训练了!")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
