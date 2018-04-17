<?php
require 'vendor/autoload.php';

function sendCurlRequest($url,$urlFields=array()){

         $headers = array();
         $postData = array();

         $fields_string = "";

         foreach ($urlFields as $key => $value) {
             $fields_string .= $key.'='.$value.'&';
         }

         $ch = curl_init();
         curl_setopt($ch, CURLOPT_URL,$url);
         curl_setopt($ch, CURLOPT_FAILONERROR,1);
         curl_setopt($ch, CURLOPT_FOLLOWLOCATION,1);
         curl_setopt($ch, CURLOPT_RETURNTRANSFER,1);
         curl_setopt($ch, CURLOPT_TIMEOUT, 15);
         curl_setopt($ch,CURLOPT_POST, count($urlFields));
         curl_setopt($ch,CURLOPT_POSTFIELDS, $fields_string);

         // Add Headers in Request
         if(!empty($headers)){
             curl_setopt($ch,CURLOPT_HTTPHEADER,$headers);
         }

         // Add Post Data to Request(if any)
         if(!empty($postData)){
           $field_string = http_build_query($postData);
           curl_setopt($ch, CURLOPT_POST, 1);
           curl_setopt($ch, CURLOPT_POSTFIELDS, $field_string);
         }

         $result = curl_exec($ch);
         return $result;

     }
$LISTING='LISTING';
// $LISTING='EXAM';
// $LISTING='COMPARISON';
$BASE_DIR="/home/mohd/shiksha_repo/shikshaoss/online_models/auto_ans";
$BOOST="^3";
$inputFile='';
$outputFile='';

switch ($LISTING) {
    case 'LISTING':
        $inputFile = $BASE_DIR."/Auto answering - Listing_2.1.xlsx";
        $outputFile = fopen($BASE_DIR.'/listing_q.csv', 'w');
        break;
    case 'EXAM':
        $inputFile = $BASE_DIR."/Auto answering - Exam_2.1.xlsx";
        $outputFile = fopen($BASE_DIR.'/exam_q.csv', 'w');
        break;
    case 'COMPARISON':
        $inputFile = $BASE_DIR."/Comparison classes.xlsx";
        $outputFile = fopen($BASE_DIR.'/comparison_q.csv', 'w');
        break;
    case 'CATEGORY':
        break;
    default:
        break;
}


fputcsv($outputFile, array('question_id', 'questions', 'class', 'direct_indirect', 'factual_opinion', 'is_original'));

// print_r($inputFile);
use PhpOffice\PhpSpreadsheet\Spreadsheet;
use PhpOffice\PhpSpreadsheet\Reader\Xlsx;

$objReader = new PhpOffice\PhpSpreadsheet\Reader\Xlsx();
$objReader->setLoadAllSheets();
$objPHPExcel = $objReader->load($inputFile);
$loadedSheetNames = $objPHPExcel->getSheetNames();
$column = array();
$final  = array(array());
$direct  = array(array());
$opinion  = array(array());

foreach($loadedSheetNames as $sheetIndex => $loadedSheetName) {
    if(!in_array($loadedSheetName,array('Sheet2','Sheet1'))){
        $highestRow = $objPHPExcel->getSheet($sheetIndex)->getHighestRow(); 
        $highestColumn = $objPHPExcel->getSheet($sheetIndex)->getHighestColumn();

        $sheetData = $objPHPExcel->getSheet($sheetIndex)->rangetoArray('A2:' . $highestColumn . $highestRow,NULL,TRUE,FALSE);
        $maxRow=-1;
        foreach ($sheetData as $key => $columnName) { 
            foreach ($columnName as $columnIndex => $value) {
                
                if(!empty($value)){
                    
                    $column[$loadedSheetName][$key][$columnIndex] = $value;
                    $maxRow=$key;
                }
            }
        }
        // a($column);die();
        $q = array();
        $d = array();
        $o = array();
        for ($i = 0;$i <= $maxRow;$i++)
        {
            $q[$i] = $column[$loadedSheetName][$i][1]; 
            
            // change the 2 to 4 in the next 2 lines for exams file.
            $label_index=2;
            if($LISTING=='EXAM'){$label_index=4;}

            $o[$i] = (strpos(strtolower($column[$loadedSheetName][$i][$label_index]),'factual')!==FALSE)?'factual':'opinion';
            $d[$i] = (strpos(strtolower($column[$loadedSheetName][$i][$label_index]),'indirect')!==FALSE)?'indirect':'direct';
            $my ='';
            for($j = 0 ; $j < strlen($q[$i]); $j++)
            {
          
                if($q[$i][$j] == "$")
                {

                    $my = '';

                    $j = $j + 2;
                    while($q[$i][$j] != "$")
                    {
                        $my .= $q[$i][$j];
                        $j++;
                    }
                    $j = $j + 2;
                    break;
                }    
            }
            $r = $my;
            $my = "$$".$my."$$";
            $r = $r.$BOOST;
            $q[$i] = str_replace($my, $r, $q[$i]);
         }
         //a($o);
        $final[$loadedSheetName] = $q;
        $direct[$loadedSheetName] = $d;
        $opinion[$loadedSheetName] = $o;
    }
}

$final = array_filter($final);
$direct = array_filter($direct);
$opinion = array_filter($opinion);

// print_r($opinion);
// print_r($direct);
// print_r($final);
// die();

foreach ($final as $sectionName => $sectionData) {
    print_r($sectionName);
    parseSectionData($sectionData,$direct[$sectionName],$opinion[$sectionName],$outputFile,$sectionName,$custom=array('BOOST'=>$BOOST));
    //a($direct[$sectionName]);
}

function parseSectionData($questions,$dir,$op,$file,$sectionName,$custom){
    
    foreach ($questions as $key => $value) {
        $value = str_replace(":", "\:", $value);
        $response = array();
        $r = sendCurlRequest("http://172.16.3.111:8983/solr/collection1/select?q=aa_question_title%3A(".urlencode($value).")%0AOR%0Aaa_question_title_edgeNGram%3A(".urlencode($value).")&fq=facetype%3Aaa_question&rows=400&fl=question_title%2Cscore%2Cquestion_id&wt=json&indent=true");
        $parse_r = json_decode($r, true);
        // sleep(5);
        fputcsv($file, array('0', str_replace($custom['BOOST'],"",$value), $sectionName, $dir[$key], $op[$key],'1'));
        $question_array = $parse_r["response"]["docs"];
        foreach($question_array as $i => $single_ques_arr) {
            $ques_id = $single_ques_arr["question_id"];
            $ques = $single_ques_arr["question_title"];
            $list = array($ques_id,$ques,$sectionName,$dir[$key],$op[$key],'0');
            array_push($response,$list);
            
        }
        foreach ($response as $key => $row)
        {
            // a($response);
            fputcsv($file, $row);
        }
        print_r(str_replace($custom['BOOST'],"",$value));

        echo "\n";
        echo "\n";
        echo "\n";
    }
}

fclose($outputFile);
function a($value='')
 {
    echo "<pre>";
    print_r($value);
    echo "</pre>";
 }
?>