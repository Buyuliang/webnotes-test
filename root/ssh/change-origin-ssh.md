# change-origin-ssh

## 1️⃣ 查看当前远程地址



```bash
git remote -v
```


## 2️⃣ 修改远程地址为 SSH

```bash
git remote set-url origin git@github.com:Buyuliang/xxx.git
```


## 3️⃣ 验证修改是否成功

```bash
git remote -v
```

## 4️⃣ 测试 SSH 连接
```bash
ssh -T git@github.com
```

```bash
预期回显
Hi Buyuliang! You've successfully authenticated, but GitHub does not provide shell access.
```





