
# Tweedie plot -----------------------------------------------------------------

import tweedie
import seaborn as sns
import matplotlib.pyplot as plt

tvs = tweedie.tweedie(mu=1, p=1.3, phi=2).rvs(1000)

sns.histplot(tvs)
plt.title('Simulated Tweedie Distribution')
plt.show()