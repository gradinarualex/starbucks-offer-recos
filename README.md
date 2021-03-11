# **Starbucks Offer Model**

**Model Background**  

The Starbucks dataset used here was originally used as a take-home assignment provided by Starbucks for their job candidates.  
The data for this exercise consists of about 120,000 data points split in a 2:1 ratio among training and test files. In the experiment simulated by the data, an advertising promotion was tested to see if it would bring more customers to purchase a specific product priced at $10. Since it costs the company 0.15 to send out each promotion, it would be best to limit that promotion only to those that are most receptive to the promotion. Each data point includes one column indicating whether or not an individual was sent a promotion for the product, and one column indicating whether or not that individual eventually purchased that product. Each individual also has seven additional features associated with them, which are provided abstractly as V1-V7.

**Optimization Strategy**

The task is to use the training data to understand what patterns in V1-V7 indicate that a promotion should be provided to a user. Specifically, the goal is to maximize the following metrics:
* **Incremental Response Rate (IRR)**

IRR depicts how many more customers purchased the product with the promotion, as compared to if they didn't receive the promotion. Mathematically, it's the ratio of the number of purchasers in the promotion group to the total number of customers in the purchasers group (treatment) minus the ratio of the number of purchasers in the non-promotional group to the total number of customers in the non-promotional group (control).

<img src="https://render.githubusercontent.com/render/math?math=IRR = \frac{purch_{treat}}{cust_{treat}} - \frac{purch_{ctrl}}{cust_{ctrl}}">

* **Net Incremental Revenue (NIR)**

NIR depicts how much is made (or lost) by sending out the promotion. Mathematically, this is 10 times the total number of purchasers that received the promotion minus 0.15 times the number of promotions sent out, minus 10 times the number of purchasers who were not given the promotion.

<img src="https://render.githubusercontent.com/render/math?math=NIR=(10*purch_{treat} - 0.15*cust_{treat}) - 10*purch_{ctrl}">

**How To Test Your Strategy**  
After we have a model we're comfortable with, we'll complete the `promotion_strategy` function to pass to the `test_results` function.  
Table of actual promotion vs. predicted promotion customers: 

|           | Actual |  |
|-----------|-----|----|
| **Predicted** | Yes | No |
| Yes |  I | II |
| No  | III| IV |

The metrics are only being compared for the individuals we predict should obtain the promotion – that is, quadrants I and II. Since the first set of individuals that receive the promotion (in the training set) receive it randomly, we can expect that quadrants I and II will have approximately equivalent participants.  
Comparing quadrant I to II then gives an idea of how well your promotion strategy will work in the future. We'll analyze how each variable or combination of variables along with a promotion influences the chance of purchasing.  
We'll test our strategy against the test dataset used in the final `test_results` function.

---

## **Licensing**
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details