$(function() {
    $('#full_label-selector').change(function(){
        $('.full_label').hide();
        $('.full_label.' + $(this).val()).show();
    });

    $('.partial_label-selector-3D').change(function(){
        var dataset = $('#partial_label-selector1_3D').val();
        var label = $('#partial_label-selector2_3D').val();

        $('.partial_label_3D').hide();
        $('.partial_label_3D.' + dataset + '.' + label).show();
        $('.semantic_label_3D').hide();
        $('.semantic_label_3D.' + dataset).show();
        // console.log('.partial_label_3D.' + dataset + '.' + label)
    });

    $('#live-selector').change(function(){
        $('.live').hide();
        $('.live.' + $(this).val()).show();
    });

    $('#obj3D-selector').change(function(){
        $('.obj3D').hide();
        $('.obj3D.' + $(this).val()).show();
    });
});

