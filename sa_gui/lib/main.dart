import 'dart:io';
import 'dart:ui';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

const String version = "1.0.2";    // how to get this from bundle or pubspec ?

void main() => runApp(MyNLPApp());

class MyNLPApp extends StatelessWidget {
  // This widget is the root of your application.

  @override
  Widget build(BuildContext context) {

    return MaterialApp(
      title: 'Foo Sentiment Analysis Runner',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(title: 'Foo Sentiment Analysis ($version)'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);
  final String title;
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final Firestore _firestore = Firestore.instance;

  Future<void> saveToFirestore(String text) async {
      print('save to firestore 2 on server: ' + text);
    if (text.length > 0) {
      print('save to firestore 2 ' + text);
      await _firestore.collection('queries').add({
        'text': text,
        'from': 'Anonymous',
        'date': DateTime.now().toIso8601String().toString(),
      });
    }
  }

  String getSentiment(num score) {
    // order of these is important
    if (score < -0.75) 
      return 'Very Negative';
    if (score < -0.25)
      return 'Negative';
    if (score > 0.75)
      return 'Very Positive';
    if (score > 0.25)
      return 'Positive';
    if (score >= -.25 && score <= 0.25)
      return 'Neutral';
    else
      return 'Unclear Sentiment: '+score.toStringAsFixed(4);
  }

  void _launchURL(String url) async {
    if (await canLaunch(url)) {
      await launch(url);
    } else {
      throw 'Could not launch $url';
    }
  }

  // to be replaced by source elements["url"]
  final modelUrls = {
    "Vader" : "https://pypi.org/project/vaderSentiment/",
    "TextBlob" : "https://textblob.readthedocs.io/en/dev/",
    "Azure NLP" : "https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/",
    "Google NLP" : "https://cloud.google.com/natural-language/",
  };

  String getModelUrl(String model) {
    if (modelUrls.containsKey(model)) {
      return modelUrls[model];
    }
    else {
      return "http://foostack.ai";
    }
  }

  void _showHelp() {
    print('showHelp clicked');
    showDialog(context: context, builder: (_) => AlertDialog(
        title: new Text("About Foo NLP SA"),
        content: new Text("Foo NLP Sentiment Analyzer combines sentiment scores from multiple " +
        "engines to provide a cross comparison of NLP Sentiment Analysis models including our " +
        "own proprietary and top secret Foo SA model.  Click on the individual model links in results for " +
        "more information on the specific model parameters and design."),
      )
    );

    setState(() {});
  }

  void _submit() async {
    print('submit clicked');
    // need to swap out hostname for dev cycle or make parametized
    // var host = 'http://10.0.2.2:5000/nlp/sa/all';   // for android emulator
    //var host = 'http://127.0.0.1:5000/nlp/sa/all';   // for web testing
    //    var host = 'https://foonlpfunc.azurewebsites.net/api/SA';   // azure serverless
    var host = 'https://flaskmli3.azurewebsites.net/nlp/sa/all';
    var response = await http.get(host+'?data='+inputController.text);
    print('Response status: ${response.statusCode}');
    print('Response body: ${response.body}');
    print(inputController.text);

    var resp = response.body;
    Map<String, dynamic> nlps = jsonDecode(resp);
    List<dynamic> results = nlps['results'];

    List<Map<String, dynamic>> dataList = [];

    results.forEach((result) {
      dataList.add(result);
    });

    saveToFirestore(inputController.text);

    rawContentList.insert(0,[dataList, inputController.text]);   // hack to add tuple for now to pass search text
    inputController.text = '';

//    FocusScope.of(context).requestFocus(FocusNode());
    FocusScope.of(context).unfocus();
    setState(() {
    });
  }

  final inputController = TextEditingController();
  final List rawContentList = [];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: ListView(
        children: <Widget>[
        Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: <Widget>[
              Padding(padding: EdgeInsets.only(top: 20.0)),
              Text('Foo Sentiment Multi Analyzer',
                style: new TextStyle(color: Colors.blue, fontSize: 25.0),),
              Padding(padding: EdgeInsets.only(top: 25.0)),
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: ConstrainedBox(
                constraints: BoxConstraints(maxWidth: 600),
                  child: TextFormField(
                    controller: inputController,
                    maxLines: 3,
                    onFieldSubmitted: (term) {
                      FocusScope.of(context).unfocus();
//                      _submit();   // causes a double submit
//                      FocusScope.of(context).requestFocus(FocusNode());
                    },
                    textInputAction: TextInputAction.done,
                    decoration: new InputDecoration(
                      labelText: "New Text to Analyze",
                      fillColor: Colors.white,
                      border: new OutlineInputBorder(
                        borderRadius: new BorderRadius.circular(25.0),
                        borderSide: new BorderSide(
                        ),
                      ),
                      //fillColor: Colors.green
                    )
                  ),
                ),
              ),
               FlatButton (
                child: Text('Go'),
                color: Colors.blue,
                textColor: Colors.white,
                onPressed: _submit,
              ),
              Divider(),
              Padding(padding: EdgeInsets.only(top: 22.0)),
              GestureDetector(
                onLongPress: () => _launchURL('https://flaskmli3.azurewebsites.net/nlp/history'),
                onTap: () => _launchURL('https://flaskmli3.azurewebsites.net/nlp/history'),
                child:
                  Text('Results',style: new TextStyle(color: Colors.blue, fontSize: 15.0),),
              ),
              Wrap(
                direction: Axis.horizontal,
                children: rawContentList.asMap().map((index, item) => MapEntry(index, buildContainer(item[0],item[1], index))).values.toList(),
              )
            ],
          ),
        ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _showHelp,
        tooltip: 'Help',
        child: Icon(Icons.help),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }

  Container buildContainer(List<Map<String, dynamic>> resultList, String inputText, num index) {
    return Container(
      width: 400,
      margin: new EdgeInsets.all(10.0),
      color: index == 0 ? Colors.green[50] : Colors.grey[100],
      child: Column (
        children: <Widget>[
          DataTable(
            columns: [
              DataColumn(label: Text('Model', style: new TextStyle(fontWeight: FontWeight.bold, color:Colors.blue, fontSize: 12.0), )),
              DataColumn(label: Text('Score', style: new TextStyle(fontWeight: FontWeight.bold, color:Colors.blue, fontSize: 12.0),)),
              DataColumn(label: Text('Sentiment', style: new TextStyle(fontWeight: FontWeight.bold, color:Colors.blue, fontSize: 12.0),)),
            ],
            rows:
            resultList // Loops through dataColumnText, each iteration assigning the value to element
                .map(((element) => DataRow(
              cells: <DataCell>[
                DataCell(
                  GestureDetector(
                    child: Tooltip(
                      message: 'Click for model details',
                        verticalOffset: 12,
                        height: 24,
                        child: Text(element["model"], style: TextStyle(decoration: TextDecoration.underline, color: Colors.blue))),
                    onTap: () => _launchURL(getModelUrl(element["model"]))
                  )
                ), //Extracting from Map element the value
                DataCell(Text(element["nScore"].toStringAsFixed(4))),
                DataCell(Text(getSentiment(element["nScore"]))),
              ],
            )),
            ).toList(),
          ),
          Text('Evaluated: ',  style: new TextStyle(color: Colors.orange, fontWeight: FontWeight.bold, fontSize: 15.0),),
          Text(inputText, style: new TextStyle(color: Colors.green, fontStyle: FontStyle.italic, fontSize: 15.0),),
        ],
      ),
    );
  }
}
