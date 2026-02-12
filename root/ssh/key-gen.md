# key-gen

## 一、配置用户名邮箱




```bash
  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"
```

## 二、生成密钥

```bash
ssh-keygen -t ed25519 -C "1259233520@qq.com"
```

## 三、获取公钥

```bash
cat C:\Users\Administrator/.ssh/id_ed25519.pub
```

## 四、将公钥提交到 code







