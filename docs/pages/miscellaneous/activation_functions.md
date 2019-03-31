Here is the list of all the activation functions currently available:

| Function 		 |   Representation |
|----------------|------------------|
| Atan           |	\begin{equation} f(x) = \text{atan}(x) \end{equation}				|  
| BentIdentity   |	\begin{equation} f(x) = \frac{\sqrt{x^2 + 1} -1 }{2}+ x \end{equation}			|          
| BipolarSigmoid |	\begin{equation} f(x) = \frac{1- e^{-x}}{1+ e^x}	\end{equation}		|            
| ELU            |	\begin{equation} f(x) = \begin{cases} x, & \text{if } x > 0 \\ \alpha (e^x -1), & \text{if } x \leq 0 \\\end{cases} \end{equation}|  
| Gaussian       |	\begin{equation} f(x) = \exp(-x^2)	\end{equation}		|        
| Hardtanh       |	\begin{equation} f(x) = \begin{cases} +1, & \text{if } x > 1 \\ -1, & \text{if } x < 1 \\ x, & \text{otherwise}  \\\end{cases} \end{equation}	| 
| Identity       |	\begin{equation} f(x) = x	\end{equation}		|        
| InverseSqrt    |	\begin{equation} f(x) = \frac{x}{ \sqrt{1 + \alpha x^2}}	\end{equation}		|          
| LeakyReLU      |	\begin{equation} f(x) = \begin{cases} x, & \text{if } x > 0 \\ 0.01x, & \text{if } x \leq 0 \\\end{cases} \end{equation}|   
| LeCunTanh      |	\begin{equation} f(x) = 1.7159 \tanh \left( \frac{2}{3} x \right)	\end{equation}		|       
| LogLog         |	\begin{equation} f(x) = 1 -\exp\left( - \exp(x) \right)	\end{equation}		|     
| LogSigmoid     |	\begin{equation} f(x) = \log\left(  \frac{1}{1 + \exp(-x)} \right) \end{equation}		|         
| ReLU           |	\begin{equation} f(x) = \begin{cases} x, & \text{if } x > 0 \\ 0, & \text{if } x \leq 0 \\\end{cases} \end{equation}|   
| SELU           |	\begin{equation} f(x) = 1.0507 \times \begin{cases} x, & \text{if } x > 0 \\ 1.67326(e^x -1), & \text{if } x \leq 0 \\\end{cases} \end{equation}| 
| Sigmoid        |	\begin{equation} f(x) = \frac{1}{1 + \exp(-x)}  \end{equation}		|     
| Sinc           |	\begin{equation} f(x) = \begin{cases} \frac{sin(x)}{x}, & \text{if } x \neq 0 \\ 1, & \text{if } x = 0 \\\end{cases} \end{equation}| 
| Softmax        |	\begin{equation} f( \vec{x} ) = \left[ \frac{\exp(x_1)}{\sum_{k=1}^K \exp(x_k)}, \frac{\exp(x_2)}{\sum_{k=1}^K \exp(x_k)}, ... ,\frac{\exp(x_K)}{\sum_{k=1}^K \exp(x_k)} \right] \end{equation}	with $\vec{x} = [x_1, x_2, ..., x_K] $	|   
| Softplus       |	\begin{equation} f(x) = \log\left( 1 + \exp(x) \right) \end{equation}		|         
| Softsign       |	\begin{equation} f(x) = \frac{x}{1 + \mid x \mid} \end{equation}		|       
| Swish          |	\begin{equation} f(x) =  \frac{x}{ 1 + \exp(-x)}  \end{equation}| 
| Tanh           |	\begin{equation} f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \end{equation}				|   


