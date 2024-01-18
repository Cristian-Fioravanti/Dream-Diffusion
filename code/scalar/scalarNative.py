# import torch
# from torch._six import inf


# class NativeScalerWithGradNormCount:
#     state_dict_key = "amp_scaler"

#     # fornisce strumenti per eseguire addestramento di reti neurali con precisione mista automaticamente. 
#     #La classe GradScaler in AMP è progettata per facilitare l'uso della precisione mista.
#     # La precisione mista è una tecnica in cui alcune parti del modello vengono eseguite in precisione ridotta 
#     #(ad esempio, FP16) per migliorare l'efficienza computazionale, mentre altre parti rimangono in precisione completa
#     # (ad esempio, FP32) per mantenere la stabilità numerica.
#     def __init__(self):
#         self._scaler = (
#             torch.cuda.amp.GradScaler()
#         ) 
        

#     def __call__(
#         self,
#         loss,
#         optimizer,
#         clip_grad=None,
#         parameters=None,
#         create_graph=False,
#         update_grad=True,
#     ):  
#         # Poiché i gradienti sono stati scalati durante il forward pass, il backward pass si basa su questi gradienti scalati. 
#         #In sostanza, è qui che i gradienti vengono effettivamente calcolati e modificati
#         # (Eseguire lo scale è utile quando si lavora con precisione ridotta per evitare problemi di overflow)
#         loss.backward(create_graph=create_graph)   
#         if update_grad:
#             if clip_grad is not None:
#                 assert parameters is not None
#                 # radienti vengono normalizzati rispetto al valore massimo clip_grad (0.8)
#                 norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad) 
#             else:
               
#                 norm = get_grad_norm_(parameters)  # norma dei gradienti senza clip
            
#             self._scaler.step(optimizer)  # esegue l'aggiornamento degli ottimizzatori applicando gli step del gradiente scalati
#             self._scaler.update()  # aggiorna la scala interna di GradScaler in base ai risultati del passo precedente
#         else:
#             norm = None
#         return norm  # restituisce la norma dei gradienti (clip o non clip), che può essere utilizzata per monitorare la stabilità del training

#     def state_dict(self):
#         return self._scaler.state_dict()

#     def load_state_dict(self, state_dict):
#         self._scaler.load_state_dict(state_dict)


# def get_grad_norm_(parameters, norm_type: float = 2.0):
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     parameters = [p for p in parameters if p.grad is not None]
#     norm_type = float(norm_type)
#     if len(parameters) == 0:
#         return torch.tensor(0.0)
#     device = parameters[0].grad.device
#     if norm_type == inf:
#         total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
#     else:
#         total_norm = torch.norm(
#             torch.stack(
#                 [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
#             ),
#             norm_type,
#         )
#     return total_norm
