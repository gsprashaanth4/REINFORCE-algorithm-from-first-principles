## REINFORCE algorithm implementation

This is a educational project, writing the REINFORCE algorithm from first principles, using my custom neural network framework. The implementation includes:

- episodic trajectory collection
- discounted return calculation
- return normalization
- categorical policy for discrete action spaces
- Gaussian policy for continuous action spaces
- manually derived policy gradients
- back propagation and optimizer updates

### Demo

REINFORCE algorithm trained on discrete CartPole-v1:
![REIN-disc](https://github.com/gsprashaanth4/REINFORCE-algorithm-from-first-principles/blob/main/media/REIN_CP.gif)

REINFORCE algorithm trained on continuous InverterPendulum-v5 only with only Mean:
![REIN-cont-m](https://github.com/gsprashaanth4/REINFORCE-algorithm-from-first-principles/blob/main/media/REIN_IP_M.gif)

REINFORCE algorithm trained on continuous InverterPendulum-v5 only with Mean and Standard-deviation:
![REIN-cont-ms](https://github.com/gsprashaanth4/REINFORCE-algorithm-from-first-principles/blob/main/media/REIN_IP_MaS.gif)