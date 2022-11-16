

import os

# shifts = ['orig', 'small_gn_shift','medium_gn_shift','large_gn_shift','small_image_shift','medium_image_shift','large_image_shift','adversarial_shift','ko_shift']
data_set = ["mnist", "cifar10" ]
shifts = ['small_image_shift','medium_image_shift','large_image_shift']
#test_types = ["univ", "multiv" ]

command = "python3.8 pipeline_landmark.py {} {} multiv"






for data in data_set:
    for shift in shifts:
        #for test_type in test_types: 
        cur_command = command.format(data, shift) 
        print(cur_command)
        # cur_command == "python3 pipline.py mnist small_image_shift multiv"
        os.system(cur_command)
        # os.system("ls")
        
