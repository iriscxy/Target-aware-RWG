import pdb

from pyrouge import Rouge155
import logging
import sys
name='result'
r = Rouge155('pyrouge/rouge/tools/ROUGE-1.5.5')


r.system_dir = name+'/decoded'
r.model_dir = name+'/reference'
r.system_filename_pattern = '(\d+)_decoded.txt'
r.model_filename_pattern = '#ID#_reference.txt'
logging.getLogger('global').setLevel(logging.WARNING)

output = r.convert_and_evaluate()
output_dict = r.output_to_dict(output)
print(str(output_dict['rouge_1_f_score']))
print(str(output_dict['rouge_2_f_score']))
print(str(output_dict['rouge_l_f_score']))
print(str(output_dict['rouge_su*_f_score']))
