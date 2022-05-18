<h1 align="center">MMLab Trading: An Algorithmic Trading Framework</h1>
**MMLab Trading is still in Beta, meaning it should be used very cautiously if used in production, as it may contain bugs.**
<p align="center">
  <img width="700" align="center" src="demo/demo.gif" alt="Logo MMLab"/>
</p>

<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://twitter.com/manhkhanhad" target="_blank">
    <img alt="Twitter: manhkhanhad" src="https://img.shields.io/twitter/follow/manhkhanhad.svg?style=social" />
  </a>
</p>



MMLab Trading is an open source Python framework apply reinforcement learning for automatic stock trading and Linear Programming for portflio optimization. 

# 🔧 Install 

# ✨ Getting Started
## Portfolio Optimization
1. Download stock historical prices and Market Index (i.e VNIndex) into `PortfolioOptimization/Data`
2. Setting the hyperparameter in `PortfolioOptimization/config_LP.yaml`
### Evaluate portfolio optimization
 ```
 python LinearOptimization.py
 ```
 
### Running


4. Train the Senario Classifier for determining whether the current is good time to start inverst
```
python SenarioClassification.py
```
5. Optimize the stock portfolio
```
python app.py
```


# 👤 Author

**Manh-Khanh Ngo Huu** [manhkhanhad](https://github.com/manhkhanhad)


# 🔰 References

#  Acknowledgement

Give a ⭐️ if this project helped you!
