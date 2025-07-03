### 问题 4
通过智能优化算法求解拟合疲劳曲线问题

本题中采用粒子群算法进行优化求解 :

参数设置如下 : 
- 粒子群大小 : 500
- 迭代次数 : 1000

```python
best_p, best_err = pso_algorithm(yp,xp,
                                     lim_min,
                                     lim_max,
                                     inertia=0.8,
                                     c1=2,
                                     c2=2,
                                     swarm_size=500,
                                     epochs=1000) 
```

得到最终各个参数的拟合结果为 : 

```shell
Best Parameters: [ 0.0086986  -0.00692268  1.         -0.86885857]  
Best Error: 0.0020 # 实际为 0.002042 左右 
```
即由粒子群方法得到疲劳曲线拟合公式为 : 
$$
x = 0.0086986 \times y^{-0.00692268 } + 1 \times y^{-0.86885857}  
$$

绘制拟合曲线如下 : 
![result.png](result.png)
